import pandas as pd
import os
import tempfile
import json
import google.generativeai as genai
import io
import numpy as np
from datetime import datetime, timezone
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file, session
from werkzeug.utils import secure_filename 
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from supabase_client import supabase
import math
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# model configuration
genai.configure(api_key=os.getenv("API_KEY"))
# model = genai.GenerativeModel('gemini-2.5-flash')
model = genai.GenerativeModel('gemini-2.5-pro')
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# backend configuration
# DEFAULT_BACKEND = "supabase" 
DEFAULT_BACKEND = "postgres"

# threshold for family name similarity (If cosine similarity between item embedding and family embedding is above this, consider it a match)
threshold_similarity = 0.82

# Database configuration 
DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "grocery_receipt_group_db"),
    "user": os.getenv("PGUSER", "receipt_user_test"),
    "password": os.getenv("PGPASSWORD", "DADE1234"),
    "host": os.getenv("PGHOST", "localhost"), 
    "port": os.getenv("PGPORT", "5432"),
}

# local postgres connection
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# define database schema
def init_db():
    with get_conn() as conn:
        with conn.cursor() as cursor:
            # extensions for 
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    company_name TEXT,
                    purchase_date DATE,
                    total_before_tax NUMERIC(12,2),
                    taxes NUMERIC(12,2),
                    total_after_tax NUMERIC(12,2),
                    total_discount NUMERIC(12,2),
                    upload_date TIMESTAMPTZ NOT NULL,
                    total_after_tax_override BOOLEAN NOT NULL DEFAULT FALSE
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS receipt_items (
                    id BIGSERIAL PRIMARY KEY,
                    receipt_id UUID NOT NULL REFERENCES receipts(receipt_id) ON DELETE CASCADE,
                    description TEXT,
                    quantity NUMERIC(12,3),
                    unit_price NUMERIC(12,2),
                    discount TEXT,
                    original_price NUMERIC(12,2),
                    discounted_price NUMERIC(12,2),
                    unit_price_after_discount NUMERIC(12,4),
                    total_price NUMERIC(12,2),
                    family_name TEXT
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS family_names (
                    family_name_id UUID PRIMARY KEY, 
                    family_name TEXT UNIQUE NOT NULL,  
                    family_embedding jsonb,          
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );       
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_user_date ON receipts(user_id, purchase_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_user_company ON receipts(user_id, company_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_receipt_id ON receipt_items(receipt_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_desc_trgm ON receipt_items USING gin (description gin_trgm_ops);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_family_name ON receipt_items(family_name);")

        conn.commit()

# ensure user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
  
# FUNCTIONS FOR EMBEDDINGS AND COSINE SIMILARITY
def preprocess_item_name(item_name):
    # Convert to lowercase
    item_name = item_name.lower()
    # Remove special characters and extra spaces
    item_name = re.sub(r'[^\w\s]', '', item_name)
    item_name = re.sub(r'\s+', ' ', item_name).strip()
    return item_name

def get_embedding(item_description):
    # create embedding using the text model
    embedding = text_model.encode(item_description)
    # ensure embedding is a numpy array
    embedding = np.array(embedding, dtype=np.float32)

    # Flatten then reshape to (1, d)
    embedding = embedding.reshape(-1)  
    embedding = embedding.reshape(1, -1)
    return embedding

def to_numpy_embedding(raw_embedding):
    # Convert various types of raw_embedding to numpy array
    if isinstance(raw_embedding, np.ndarray):
        arr = raw_embedding
    elif isinstance(raw_embedding, list):
        arr = np.array(raw_embedding, dtype=np.float32)
    elif isinstance(raw_embedding, str):
        arr = np.array(json.loads(raw_embedding), dtype=np.float32)
    else:
        raise ValueError(f"Unexpected embedding type: {type(raw_embedding)}")

    # Flatten then reshape to (1, d) (check)
    arr = arr.reshape(-1)  
    arr = arr.reshape(1, -1)
    return arr

def get_family_name(item_description, existing_families, cur=None, db_type=None, supabase=None, threshold_similarity=threshold_similarity):
   #Normalize once & embed once
    raw_desc = item_description or ""
    norm_desc = preprocess_item_name(raw_desc)
    item_vec = get_embedding(norm_desc)

    # Cache: normalized description is family_name
    hit = existing_families.get(norm_desc)
    if hit:
        return hit

    best_family_name = norm_desc  # default if no match

    # postgres path (db_type is "postgres" and cur is provided)
    if db_type == "postgres" and cur:
        # check for exact family 
        cur.execute("SELECT family_name FROM family_names WHERE family_name = %s", (norm_desc,))
        row = cur.fetchone()
        if row:
            best_family_name = row[0]
        else:
            # check for similarity with existing families
            cur.execute("SELECT family_name, family_embedding FROM family_names")
            for fam_name, fam_emb in cur.fetchall():
                if not fam_emb:
                    continue
                # Ensure family embedding is a numpy array
                fam_vec = to_numpy_embedding(fam_emb)  # (1, d)
                # Compute cosine similarity
                sim = cosine_similarity(item_vec, fam_vec)[0][0]  # scalar
                if sim >= threshold_similarity:
                    # Found a match
                    best_family_name = fam_name
                    break

            # if no match create new family (store normalized)
            if best_family_name == norm_desc:
                cur.execute(
                    """
                    INSERT INTO family_names (family_name_id, family_name, family_embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (family_name) DO NOTHING
                    """,
                    (str(uuid.uuid4()), norm_desc, psycopg2.extras.Json(item_vec.reshape(-1).tolist()))
                )
                #use the canonical value from DB to handles concurrent conflict
                cur.execute("SELECT family_name FROM family_names WHERE family_name = %s", (norm_desc,))
                row2 = cur.fetchone()
                best_family_name = row2[0] if row2 else norm_desc

    # supabase path (db_type is "supabase" and supabase client is provided)
    elif db_type == "supabase" and supabase:
        # Check for exact family name
        # existing_family = supabase.table("family_names").select("family_name").eq("family_name", item_description).execute().data
        existing_family = supabase.table("family_names").select("family_name").eq("family_name", norm_desc).execute().data
        
        if existing_family:
            best_family_name = existing_family[0]["family_name"]
        else:
            # check for similarity with existing families
            existing_family = supabase.table("family_names").select("family_name_id", "family_name", "family_embedding").execute().data
            for family in existing_family:
                # get family name and embedding den ensure it's a numpy array
                family_name = family["family_name"]
                family_embedding = to_numpy_embedding(family["family_embedding"])
                # Compute cosine similarity
                similarity = cosine_similarity(item_vec, family_embedding)[0][0]
                if similarity >= threshold_similarity:
                    # Found a match
                    best_family_name = family_name
                    break
            # if no match create new family (store normalised)
            if best_family_name == norm_desc:
                family_name_id = str(uuid.uuid4())
                supabase.table("family_names").insert({
                    "family_name_id": family_name_id,
                    "family_name": norm_desc,
                    "family_embedding": item_vec.reshape(-1).tolist()
                }).execute()

    # add created family name to this temporary dict
    existing_families[norm_desc] = best_family_name
    return best_family_name

# FUNCTIONS FOR INSERTING RECEIPTS AND ITEMS
def _insert_postgres(receipt, user_id, conn=None):
    # Create unique receipt ID
    receipt_id = str(uuid.uuid4())
    cur = conn.cursor()

    # Insert receipt into the database
    cur.execute(
        """
        INSERT INTO receipts (
            receipt_id, user_id, company_name, purchase_date,
            total_before_tax, taxes, total_after_tax,
            total_discount, upload_date
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            receipt_id,
            user_id,
            receipt.get("company_name", ""),  
            receipt.get("purchase_date", None),  
            float(receipt.get("total_before_tax") or 0),
            float(receipt.get("taxes") or 0),  
            float(receipt.get("total_after_tax") or 0), 
            float(receipt.get("total_discount") or 0),  
            datetime.now()  # Use current timestamp as upload date
        )
    )

    # Prepare list of items to insert
    items_to_insert = []
    # cache to hold family name mappings for this file (cache and adding happens in get_family name function)
    cache = {} 
    for item in receipt.get("items", []):
        try:
            # Extract item details like quantity, unit price, and discount
            quantity = float(item.get("quantity", 0) or 0)
            unit_price = float(item.get("unit_price", 0) or 0)
            discount = item.get("discount", "") or ""
            original_price = round(quantity * unit_price, 2)  # Calculate original price
            source = receipt.get("source", "upload")  # Get source of receipt
            discount_amt = compute_discount(discount, quantity, unit_price, source)  # Compute discount amount
            discounted_price = round(original_price - discount_amt, 2)  # Apply discount
            unit_price_after_discount = round(discounted_price / quantity, 2) if quantity else 0.0  # Calculate price after discount

            # Find family name for this item
            family_name = get_family_name(
                item.get("description", "") or "",  # Get item description or empty string
                existing_families=cache,  # Use cache for faster lookups
                cur=cur,
                db_type="postgres"  # Use PostgreSQL backend
            )

            # Add item details to the list to insert
            items_to_insert.append((
                receipt_id,
                item.get("description", "") or "",
                quantity,
                unit_price,
                discount,
                original_price,
                discounted_price,
                unit_price_after_discount,
                discounted_price,
                family_name  # Store the family name
            ))
        except Exception as e:
            print(f"[Item Skipped] {e} - {item}")  # If an error occurs with an item, log it and skip

    # Insert all items into the database in one operation
    if items_to_insert:
        cur.executemany(
            """
            INSERT INTO receipt_items (
                receipt_id, description, quantity, unit_price,
                discount, original_price, discounted_price,
                unit_price_after_discount, total_price, family_name
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            items_to_insert  # List of items to insert
        )

    # Commit the transaction and close the cursor
    conn.commit()
    cur.close()
    return receipt_id  # Return the generated receipt ID

def _insert_supabase(receipt, user_id, supabase):
    # Create unique receipt ID
    receipt_id = str(uuid.uuid4())  

    # Insert the receipt into the Supabase database
    supabase.table("receipts").insert({
        "receipt_id": receipt_id,
        "user_id": user_id,
        "company_name": receipt.get("company_name", ""),
        "purchase_date": receipt.get("purchase_date", ""),
        "total_before_tax": float(receipt.get("total_before_tax") or 0),
        "taxes": float(receipt.get("taxes") or 0),
        "total_after_tax": float(receipt.get("total_after_tax") or 0),
        "total_discount": float(receipt.get("total_discount") or 0),
        "upload_date": datetime.now().date().isoformat(),  # Use current date as upload date
    }).execute()

    # Prepare list of items and cache existing families for this file (cache and adding the name to exisiting dict happens in get_family name function)
    items = []
    existing_families = {}  

    for item in receipt.get("items", []):
        try:
            # Extract item details like quantity, unit price, and discount
            quantity = float(item.get("quantity", 0))
            unit_price = float(item.get("unit_price", 0))
            discount = item.get("discount", "")
            original_price = round(quantity * unit_price, 2)
            source = receipt.get("source", "upload")
            discount_amt = compute_discount(discount, quantity, unit_price, source)
            discounted_price = round(original_price - discount_amt, 2)
            unit_price_after_discount = round(discounted_price / quantity, 2) if quantity else 0.0

            # checks if a specific family_name already exists in the family_names table of a Supabase database.
            family_name = get_family_name(item.get("description", ""), existing_families, cur=None, db_type="supabase", supabase=supabase)

            # Insert the item into receipt_items table with the family_name
            items.append({
                "receipt_id": receipt_id,
                "description": item.get("description", ""),
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "original_price": original_price,
                "discounted_price": discounted_price,
                "unit_price_after_discount": unit_price_after_discount,
                "total_price": discounted_price,
                "family_name": family_name  # Store family name in the table
            })

        except Exception as e:
            print(f"[Item Skipped] {e} - {item}")  # If an error occurs, log and skip the item

    # Insert all items into the Supabase database
    if items:
        supabase.table("receipt_items").insert(items).execute()

    return receipt_id  # Return the generated receipt ID

def insert_receipt_and_items(receipt, user_id, supabase=None, conn=None):
    backend = DEFAULT_BACKEND
    try:
        if backend == "postgres":
            return _insert_postgres(receipt, user_id, conn=conn)
        elif backend == "supabase":
            if supabase is None:
                raise RuntimeError("Supabase client not provided.")
            return _insert_supabase(receipt, user_id, supabase)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")
    except Exception as e:
        print(f"[Insert Error] backend={backend} -> {e}")
        raise

def save_receipt_for_current_user(receipt):
    # Save the receipt for the current user
    user_id = session['user_id']
    if DEFAULT_BACKEND == "supabase":
        # Pass the insert receipt and items function and Supabase client 
        return insert_receipt_and_items(receipt, user_id=user_id, supabase=supabase)
    elif DEFAULT_BACKEND == "postgres":
        # pass the insert erceipt and items function and use a short-lived PostgreSQL connection
        with get_conn() as conn:
            return insert_receipt_and_items(receipt, user_id=user_id, conn=conn)
    else:
        raise RuntimeError(f"Unknown DEFAULT_BACKEND: {DEFAULT_BACKEND}")
    
# FUNCTIONS FOR EXTRACTING RECEIPT DATA
def extract_receipt_data(file_path):
    # Upload the file to Gemini
    uploaded_file = genai.upload_file(path=file_path)
    # prompt for gemini multimodal model to extract receipt data
    prompt_parts = [
        f"The file '{os.path.basename(file_path)}' may contain one or more receipts or invoices.",
        "Your task is to extract detailed structured data from each receipt or invoice found in the document.",

        "",
        "For **each** receipt or invoice, extract the following fields:",
        "- `company_name`: The name of the business issuing the receipt or invoice.",
        "- `date`: The date the receipt or invoice was issued. Format it as `YYYY-MM-DD`.",
        "- `items`: A list of purchased items, each with the following:",
        "    - `description`: Name or description of the item",
        "    - `quantity`: Number of units",
        "    - `unit_price`: Price per unit (the original price before any discount if available)",
        "    - `discount`: A **single numeric amount** for the item-level discount. Return a number only (e.g., -1.61 or 0).",
        "        • If the receipt shows text like `PWP Discount 2.35`, treat it as a reduction and return `-2.35`.",
        "        • If the receipt shows `PWP Discount -1.15`, return `-1.15`.",
        "        • If no item-level discount applies, return `null`.",
        "        • If multiple discount numbers appear for the same item (e.g., `2 for $3.99, -1.61`), return the **net per-item discount amount** as a single number (e.g., `-1.61`).",
        "        • Do **not** include any words like 'PWP', 'Promo', '%', 'Usual', 'Now'; return the numeric value only.",
        "    - `total_price`: Total price for this item **after** any discount (as written, do not calculate)",
        "- `taxes`: Any applicable taxes such as GST (if present)",
        "- `total_before_tax`: Subtotal before any tax or discount",
        "- `total_discount`: The total discount applied to the entire receipt (not individual items), if present. This could be a flat amount (e.g., $5 off), or a percentage.",
        "- `total_after_tax`: Final total after including tax and applying any receipt-level discounts",

        "",
        "**Important Notes**:",
        "- Make sure to **correctly match item-level discounts** to the appropriate item, even if the discount is printed on the next line or uses labels like 'Usual', 'Now', 'Promo', or 'Normal Price'.",
        "- Ignore promotional lines such as '3 for $2.95', 'PWP Discount', 'Promotion', or 'Total Savings' if they are not actual purchased items. However, use these clues if needed to deduce item-level pricing or discounts.",
        "- Do **not** extract promotions or savings banners as items.",
        "- If there's no discount, set `discount` to `null`.",
        "- Avoid performing calculations. Just extract values exactly as shown on the receipt, except normalize `discount` to a numeric amount as instructed.",
        "- Do not extract items such as 'Plastic Bag', 'Carrier Bag', 'Shopping Bag', 'BAG' or similar packaging charges. Ignore them completely."
        "",

        "Return the results as a **JSON array** of receipt objects. Each object should follow this structure:",

        "```json",
        "{",
        '  "company_name": "string",',
        '  "date": "YYYY-MM-DD",',
        '  "total_before_tax": number_or_string,',
        '  "taxes": number_or_string,',
        '  "total_discount": number_or_string,',
        '  "total_after_tax": number_or_string,',
        '  "items": [',
        "    {",
        '      "description": "string",',
        '      "quantity": number_or_string,',
        '      "unit_price": number_or_string,',
        '      "discount": number_or_null,',
        '      "total_price": number_or_string',
        "    }",
        "  ]",
        "}",
        "```",
        "",
        uploaded_file
    ]
    response = model.generate_content(prompt_parts,
        safety_settings={
            "HARASSMENT": "BLOCK_NONE",  # Block harassment content
            "HATE_SPEECH": "BLOCK_NONE",  # Block hate speech
            "SEXUAL": "BLOCK_NONE",  # Block sexual content
            "DANGEROUS": "BLOCK_NONE",  # Block dangerous content
        })
    
    # Get and clean the response text
    text = (response.text or "").strip()  
    # Extract JSON block safely
    if "```json" in text:
        # Split the text to get the part after the first "```json" marker
        text = text.split("```json", 1)[-1]
    if "```" in text:
        # Remove content after the first closing "```" marker
        text = text.split("```", 1)[0]
    text = text.strip()  # Clean up the text

    # Fall back to [] if parsing fails
    try:
        data = json.loads(text)  # Attempt to parse the JSON content
    except Exception:
        data = [] 
    # Ensure the parsed data is a list of receipts
    receipts = data if isinstance(data, list) else [data] if data else []

    # empty list for cleaned receipts in a path
    cleaned = []
    for receipt in receipts:
        # Function to coerce a value into a number, handling formatting issues like commas
        def coerce_num(x):
            try:
                # Remove commas for values like "1,234.50" and convert to float
                return float(str(x).replace(",", ""))
            except Exception:
                return x  # Return original value if conversion fails

        # empty list for the items in the receipt
        items = []
        for item in (receipt.get("items") or []):  # Iterate through items in the receipt
            q_raw = item.get("quantity", 0)  # Get the quantity
            try:
                # Convert quantity to float, handling cases like "1,234.50"
                q_num = float(str(q_raw).replace(",", ""))
            except Exception:
                q_num = 0.0  

            if q_num != 0:  # Only include item if the quantity is non-zero
                # Append item details to the items list
                items.append({
                    "description": (item.get("description") or "").strip(),  # Clean description text
                    "quantity": q_num,  # Store numeric quantity
                    "unit_price": coerce_num(item.get("unit_price")),  # Store numeric unit price
                    "total_price": coerce_num(item.get("total_price")),  # Store numeric total price
                    "discount": coerce_num(item.get("discount")) if item.get("discount") is not None else None  # Store discount if there is one
                })

        # Add cleaned receipt data to the list
        cleaned.append({
            "company_name": (receipt.get("company_name") or "").strip(),  
            "purchase_date": (receipt.get("date") or "").strip(),  
            "total_before_tax": coerce_num(receipt.get("total_before_tax")),  
            "taxes": coerce_num(receipt.get("taxes")),  # Coerce taxes
            "total_after_tax": coerce_num(receipt.get("total_after_tax")), 
            "total_discount": coerce_num(receipt.get("total_discount")),  
            "items": items, 
        })
    print("Extracted receipts:", receipts)
    return cleaned # return cleaned receipts

# Function to compute discount for an item based on various formats (percentage, currency, or plain number)
def compute_discount(discount_str, quantity, unit_price, source="upload"):
    try:
        discount_str = str(discount_str or "").strip().lower()  # Convert discount into clean strin
        if not discount_str:  
            return 0.0

        # Remove minus + parentheses and currency markers + remove commas
        s = discount_str.replace(",", "") 
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]  # Treat (1.23) as negative 1.23
        s = s.replace("-", "").replace("s$", "").replace("rm", "").replace("usd", "").strip()
        # Handle Percentage discount (e.g., "5%" or "5% off")
        # Calculate total price for the item for %
        total_price = float(quantity) * float(unit_price)
        if "%" in s:
            percent = float(s.replace("%", "").replace("off", "").strip())
            return round((percent / 100.0) * total_price, 2)

        # Handle Currency discounts (e.g., "$1.20 off" or "SGD 2.00")
        if "$" in s:
            parts = [p for p in s.replace("off", "").split() if "$" in p]
            nums = []
            for p in parts:
                try:
                    nums.append(float(p.replace("$", "")))  # Extract numeric value from currency string
                except Exception:
                    pass
            return round(sum(nums), 2) if nums else 0.0
        
        # Handle Plain Numeric Discount (no % or $)
        val = float(s.replace("off", "").strip())
        if source == "manual":
            # If val looks too big (greater than unit price), assume per-item discount and multiply by quantity.
            return round(val * float(quantity), 2) if val > float(unit_price) else round(val, 2)
        else:
            # If
            return round(val, 2)

    except Exception:
        return 0.0  # Return 0.0 if there is any error in processing the discount

# Function to update a receipt in PostgreSQL by its ID
def _pg_update_receipt_by_id(receipt_id, payload, conn=None):
    if not payload:  # If there is no payload, do nothing
        return
    owns = conn is None  # Check if the connection is passed or needs to be created
    if owns:
        conn = get_conn()  # Get a new connection if not passed
    try:
        # Prepare column names and values for the SQL query
        cols, vals = [], []
        for k, v in payload.items():
            cols.append(f"{k} = %s")  # Prepare columns to update
            vals.append(v)  # Add corresponding values
        vals.append(receipt_id)  # Add the receipt_id at the end of the values list
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE receipts SET {', '.join(cols)} WHERE receipt_id = %s",  # Dynamic column updates
                tuple(vals),  # Execute with the values as parameters
            )
        if owns: 
            conn.commit()  # Commit changes if the connection was created internally
    finally:
        if owns:
            conn.close()  # Close the connection if it was created internally

# Function to recalculate the totals of a receipt
def recalculate_receipt_totals(receipt_id, *, respect_override=True, supabase=None, conn=None):
    if DEFAULT_BACKEND == "supabase":
        # If using Supabase backend, fetch receipt data and items
        rec = (supabase.table("receipts")
               .select("taxes, total_after_tax_override")
               .eq("receipt_id", receipt_id).single().execute()).data
        if not rec: return  # Return if no record found
        taxes = float(rec.get("taxes") or 0)  # Get taxes
        override = bool(rec.get("total_after_tax_override")) if respect_override else False  # Check for override
        # Fetch receipt items from Supabase
        items = (supabase.table("receipt_items")
                 .select("original_price, discounted_price")
                 .eq("receipt_id", receipt_id).execute()).data or []

        # Calculate totals: subtotal, discounted total, and total discount
        subtotal = round(sum(float(i.get("original_price") or 0) for i in items), 2)
        sum_disc = round(sum(float(i.get("discounted_price") or 0) for i in items), 2)
        total_disc = round(subtotal - sum_disc, 2)
        computed_total = round(sum_disc + taxes, 2)

        # Prepare payload to update receipt
        payload = {"total_before_tax": subtotal, "total_discount": total_disc}
        if not override:
            payload["total_after_tax"] = computed_total  # Set total after tax if no override

        # Update receipt in Supabase
        supabase.table("receipts").update(payload).eq("receipt_id", receipt_id).execute()
        return

    # If postgre 
    owns = conn is None  # Check if the connection was passed
    if owns:
        conn = get_conn()  # Get a new connection if not passed
    try:
        #1: read taxes and override flag from postgresql
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COALESCE(taxes,0)::float AS taxes, COALESCE(total_after_tax_override,false) AS override "
                "FROM receipts WHERE receipt_id = %s LIMIT 1",
                (receipt_id,),
            )
            rec = cur.fetchone()
        if not rec:
            return  

        taxes = float(rec["taxes"] or 0.0)  # Get taxes
        override_flag = bool(rec["override"]) if respect_override else False  # check for override

        # aggregate item totals
        agg = _pg_aggregate_totals(receipt_id, conn=conn)
        if not agg:
            return  

        #prepare payload for updating the receipt
        payload = {
            "total_before_tax": agg["subtotal"],  # Subtotal before discount
            "total_discount": agg["discount_total"],  # Total discount
        }
        if not override_flag:
            payload["total_after_tax"] = agg["total_after_tax"]  # Total after tax if no override

        # 3: Update receipt totals in the database
        _pg_update_receipt_by_id(receipt_id, payload, conn=conn)

        if owns:
            conn.commit()  # commit changes if the connection was created internally
    finally:
        if owns:
            conn.close()  # close the connection if it was created internally

# --------------------------------------
# USERS FUNCTIONS
# check users table in Supabase to check if a user with the given username exists.
def _sb_get_user_by_username(username, supabase):
    res = supabase.table("users").select("*").eq("username", username).limit(1).execute()
    data = res.data or []  # If no user is found, return an empty list.
    return data[0] if data else None  # return user if found, else None.

#checks if the username already exists and, if not, creates a new user.
def _sb_create_user(username, password_hash, supabase):
    res = supabase.table("users").select("id").eq("username", username).limit(1).execute()
    if res.data:
        return None, "exists"  # Return None and a message if the username already exists.
    # Create a unique user ID and insert the new user into the users table.
    user_id = str(uuid.uuid4())
    supabase.table("users").insert({
        "id": user_id,
        "username": username,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).isoformat()  # Store the current timestamp.
    }).execute()

    return {"id": user_id, "username": username, "password_hash": password_hash}, None  # Return the created user and None for no error.

# checks users table in postgresql to fetch user data based on the provided username.
def _pg_get_user_by_username(username, conn=None):
    owns_conn = conn is None  # check if a connection is provided or if one needs to be opened.
    if owns_conn:
        conn = get_conn()  # Get a new connection if none is provided.

    try:
        if owns_conn:
            with conn:  # Open and use the connection in this block.
                with conn.cursor(cursor_factory=RealDictCursor) as cur:  # Use RealDictCursor to get results as a dictionary.
                    cur.execute(
                        "SELECT id, username, password_hash FROM users WHERE username = %s LIMIT 1",
                        (username,),
                    )
                    row = cur.fetchone()  # Fetch the first row that matches the username.
        else:
            # If a connection is passed by the caller, dont manage it here.
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = %s LIMIT 1",
                    (username,),
                )
                row = cur.fetchone()

        return dict(row) if row else None  # return user as a dictionary if found else none.
    finally:
        if owns_conn:
            conn.close()  # close the connection if it was opened by this function.

# inserts a new user into the user table in postgresql after checking if the username already exists.
def _pg_create_user(username, password_hash, conn=None):
    owns_conn = conn is None  
    if owns_conn:
        conn = get_conn()  

    try:
        if owns_conn:
            with conn:  
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", (username,))
                    if cur.fetchone():
                        return None, "exists"  # If the username exists, return None and an exist message

                    user_id = str(uuid.uuid4())  # Create a unique user ID.
                    cur.execute(
                        "INSERT INTO users (id, username, password_hash, created_at) VALUES (%s, %s, %s, NOW())",
                        (user_id, username, password_hash),  # Insert the new user into the table.
                    )
                    return {"id": user_id, "username": username, "password_hash": password_hash}, None  # Return the created user and None for no error.
        else:
            # if not the one creating the connection, just use it.
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", (username,))
                if cur.fetchone():
                    return None, "exists"  

                user_id = str(uuid.uuid4())  
                cur.execute(
                    "INSERT INTO users (id, username, password_hash, created_at) VALUES (%s, %s, %s, NOW())",
                    (user_id, username, password_hash),  
                )
            return {"id": user_id, "username": username, "password_hash": password_hash}, None 
    finally:
        if owns_conn:
            conn.close()  # Close the connection if it was opened by this function.

# Main function to retrieve a user by their username
# calls the appropriate function based on the backend selected
def get_user_by_username(username, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres":  # Check if the backend is poatgresql
        return _pg_get_user_by_username(username, conn=conn)
    elif DEFAULT_BACKEND == "supabase":  # If the backend is supabase, call the Supabase function.
        if supabase is None:
            raise RuntimeError("Supabase client not provided.") 
        return _sb_get_user_by_username(username, supabase=supabase)
    else:
        raise RuntimeError(f"Unknown backend: {DEFAULT_BACKEND}")  

# Main function to create a user with the given username and password hash
def create_user(username, password_hash, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres":  
        return _pg_create_user(username, password_hash, conn=conn)
    elif DEFAULT_BACKEND == "supabase":  
        if supabase is None:
            raise RuntimeError("Supabase client not provided.")
        return _sb_create_user(username, password_hash, supabase=supabase)
    else:
        raise RuntimeError(f"Unknown backend: {DEFAULT_BACKEND}") 

# ----------------------------------------------------
# view receipts route functions
# function to fetch receipts and its items from supabase
def _sb_fetch_receipts_and_items(user_id, supabase):
    # fetch receipts for the user, ordered by purchase date in descending order
    receipts_res = (
        supabase.table("receipts")
        .select("*")
        .eq("user_id", user_id)
        .order("purchase_date", desc=True)
        .execute()
    )
    receipts = receipts_res.data or []  # handle case where no receipts are found

    items_by_receipt = {}  # dictionary to store items grouped by receipt
    if receipts:
        receipt_ids = [r["receipt_id"] for r in receipts]  # list of receipt ids 
        # fetch items for each receipt
        items_res = (
            supabase.table("receipt_items")
            .select("*")
            .in_("receipt_id", receipt_ids)  # only fetch items for the selected receipts
            .execute()
        )
        items = items_res.data or []  # handle case where no items are found
        # group items by receipt_id
        for item in items:
            items_by_receipt.setdefault(item["receipt_id"], []).append(item)

    return receipts, items_by_receipt  # return receipts and the grouped items

# function to fetch receipts and its items from postgresql
def _pg_fetch_receipts_and_items(user_id, conn=None):
    owns = conn is None  
    if owns:
        conn = get_conn()  
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # fetch receipts for the user, ordered by purchase date in descending order
            cur.execute(
                """
                SELECT *
                FROM receipts
                WHERE user_id = %s
                ORDER BY purchase_date DESC NULLS LAST
                """,
                (user_id,),
            )
            receipts = [dict(r) for r in cur.fetchall()]  # convert results to a list of dictionaries

            items_by_receipt = {}  # dictionary to store items grouped by receipt
            if receipts:
                # fetch items associated with the receipts
                cur.execute(
                    """
                    SELECT ri.*
                    FROM receipt_items ri
                    JOIN receipts r ON r.receipt_id = ri.receipt_id
                    WHERE r.user_id = %s
                    """,
                    (user_id,),
                )
                for row in cur.fetchall():  # loop through the items and group by receipt_id
                    d = dict(row)  # convert each row to a dictionary
                    items_by_receipt.setdefault(d["receipt_id"], []).append(d)

        return receipts, items_by_receipt  # return receipts and the grouped items
    finally:
        if owns:
            conn.close()  

# main function to fetch receipts and items based on the backend
def fetch_receipts_and_items(user_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres":  
        return _pg_fetch_receipts_and_items(user_id, conn=conn)
    elif DEFAULT_BACKEND == "supabase":  
        if supabase is None:
            raise RuntimeError("Supabase client not provided.")  
        return _sb_fetch_receipts_and_items(user_id, supabase=supabase)
    else:
        raise RuntimeError(f"Unknown backend: {DEFAULT_BACKEND}")  

# --------------------------------------------------------
# receipt tables route functions
# function to fetch unique company names by with the user's receipts supabase(sb)
def _sb_get_companies(user_id, supabase):
    res = (supabase.table("receipts")
           .select("company_name")
           .eq("user_id", user_id)
           .order("company_name")
           .execute()).data or []
    return sorted({r["company_name"] for r in res if r.get("company_name")})

# function to fetch unique company names by with the user's receipts postgresql(pg)
def _pg_get_companies(user_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        sql = """
            SELECT DISTINCT company_name
            FROM receipts
            WHERE user_id = %s AND company_name IS NOT NULL
            ORDER BY company_name
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
        return [r["company_name"] for r in rows]
    finally:
        if owns:
            conn.close()

# function to fetch paginated receipts and items from sb with filters and counts
def _sb_get_receipts_page(user_id, filters, page, per_page, supabase):
    query = supabase.table("receipts").select("*").eq("user_id", user_id)
    # apply filters
    if filters.get("company_name"):
        query = query.eq("company_name", filters["company_name"])
    if filters.get("purchase_date"):
        query = query.eq("purchase_date", filters["purchase_date"])

    # count total receipts matching the filters
    cnt_q = supabase.table("receipts").select("receipt_id", count="exact").eq("user_id", user_id)
    if filters.get("company_name"):
        cnt_q = cnt_q.eq("company_name", filters["company_name"])
    if filters.get("purchase_date"):
        cnt_q = cnt_q.eq("purchase_date", filters["purchase_date"])
    total_count = cnt_q.execute().count or 0
    total_pages = math.ceil(total_count / per_page) if per_page else 1  # calculate total pages
    # fetch receipts for the current page
    start = (page - 1) * per_page
    end = start + per_page - 1
    receipts = (query.order("purchase_date", desc=True).range(start, end).execute().data) or []
    items_by_receipt = {}
    # Extract all receipt IDs from the list of receipts
    rids = [r["receipt_id"] for r in receipts]
    if rids:
        # Fetch all items link with the extracted receipt IDs from receipt_items table
        items = (supabase.table("receipt_items").select("*")
                 .in_("receipt_id", rids).execute().data) or []
        # loop through items and group them by receipt_id
        for it in items:
            items_by_receipt.setdefault(it["receipt_id"], []).append(it)
    return receipts, items_by_receipt, total_pages

# function to fetch paginated receipts and items from pg with filters and counts
def _pg_get_receipts_page(user_id, filters, page, per_page, conn=None):
    owns = conn is None 
    if owns:
        conn = get_conn() 

    where = ["user_id = %s"]
    params = [user_id]
    # apply filters
    if filters.get("company_name"):
        where.append("company_name = %s")
        params.append(filters["company_name"])
    if filters.get("purchase_date"):
        where.append("purchase_date = %s")
        params.append(filters["purchase_date"])
    where_sql = " AND ".join(where)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # count total receipts matching the filters
            cur.execute(f"SELECT COUNT(*) AS count FROM receipts WHERE {where_sql}", tuple(params))
            row = cur.fetchone()
            total_count = row["count"] if isinstance(row, dict) else row[0]
            total_pages = math.ceil(total_count / per_page) if per_page else 1

            # fetch receipts for the current page
            cur.execute(
                f"""
                SELECT *
                FROM receipts
                WHERE {where_sql}
                ORDER BY purchase_date DESC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                tuple(params + [per_page, (page - 1) * per_page]),
            )
            receipts = [dict(r) for r in cur.fetchall()]

            items_by_receipt = {}
            # Extract all receipt IDs from the list of receipts
            rids = [r["receipt_id"] for r in receipts]
            if rids:
                # Fetch all items linked with the extracted receipt IDs from receipt_items table
                cur.execute(
                    "SELECT * FROM receipt_items WHERE receipt_id = ANY(%s::uuid[])",
                    (rids,), 
                )
                # loop through items and group them by receipt_id
                for row in cur.fetchall():
                    d = dict(row)
                    items_by_receipt.setdefault(d["receipt_id"], []).append(d)

        return receipts, items_by_receipt, total_pages
    finally:
        if owns:
            conn.close()

# function to fetch a receipt for a user from sb
def _sb_get_receipt_for_user(receipt_id, user_id, supabase):
    res = (supabase.table("receipts")
           .select("receipt_id, total_after_tax, total_after_tax_override")
           .eq("receipt_id", receipt_id).eq("user_id", user_id)
           .single().execute()).data
    return res

# function to fetch a receipt for a user from pg
def _pg_get_receipt_for_user(receipt_id, user_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT receipt_id, total_after_tax, total_after_tax_override
                FROM receipts
                WHERE receipt_id = %s AND user_id = %s
                LIMIT 1
                """,
                (receipt_id, user_id),
            )
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if owns:
            conn.close()

# function to update a receipt in sb
def _sb_update_receipt(receipt_id, user_id, payload, supabase):
    (supabase.table("receipts").update(payload)
     .eq("receipt_id", receipt_id).eq("user_id", user_id).execute())

# function to update a receipt in pg
def _pg_update_receipt(receipt_id, user_id, payload, conn=None):
    if not payload:
        return

    owns = conn is None
    if owns:
        conn = get_conn()

    try:
        # Build SET clause and values
        cols = [f"{k} = %s" for k in payload.keys()]
        vals = list(payload.values())

        # Add identifiers for WHERE clause
        vals.extend([receipt_id, user_id])

        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE receipts SET {', '.join(cols)} WHERE receipt_id = %s AND user_id = %s",
                tuple(vals),
            )

        if owns:
            conn.commit()
    finally:
        if owns:
            conn.close()

# function to aggregate totals for a receipt in sb
def _sb_aggregate_totals(receipt_id, supabase):
    # fetch the taxes value for the specified receipt_id 
    rec = (supabase.table("receipts").select("taxes").eq("receipt_id", receipt_id).single().execute().data)
    if not rec: return None
    taxes = float(rec.get("taxes") or 0)
    # fetch the original and discounted prices for each item associated with the receipt
    items = (supabase.table("receipt_items").select("original_price, discounted_price")
             .eq("receipt_id", receipt_id).execute().data) or []

    #summing the original prices of all items
    subtotal = round(sum(float(i.get("original_price") or 0) for i in items), 2)
    #summing the discounted prices of all items
    sum_disc = round(sum(float(i.get("discounted_price") or 0) for i in items), 2)
    
    # dict with the calculated values: subtotal, discount, taxes, and total after tax
    return {
        "subtotal": subtotal,
        "discount_total": round(subtotal - sum_disc, 2),  
        "taxes": round(taxes, 2), 
        "total_after_tax": round(sum_disc + taxes, 2) 
    }

# function to aggregate totals for a receipt in pg
def _pg_aggregate_totals(receipt_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn() 
    
    try:
        # fetch the taxes for the specified receipt_id 
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(taxes,0)::float FROM receipts WHERE receipt_id = %s",
                (receipt_id,),  
            )
            row = cur.fetchone()  # fetch the result
            if not row:
                return None  
            
            # convert the fetched 'taxes' value to a float
            taxes = float(row[0] or 0)

            # fetch the subtotal and total discounted price for the specified receipt_id from the receipt_items table
            cur.execute(
                """
                SELECT
                  COALESCE(SUM(original_price),0)::float AS subtotal,
                  COALESCE(SUM(discounted_price),0)::float AS sum_disc
                FROM receipt_items
                WHERE receipt_id = %s
                """,
                (receipt_id,), 
            )
            # fetch the subtotal and sum of discounted prices
            subtotal, sum_disc = cur.fetchone()

        # round the subtotal and sum of discounted prices 
        subtotal = round(float(subtotal or 0), 2)
        sum_disc = round(float(sum_disc or 0), 2)
        
        return {
            "subtotal": subtotal,
            "discount_total": round(subtotal - sum_disc, 2),  
            "taxes": round(taxes, 2),  
            "total_after_tax": round(sum_disc + taxes, 2)  
        }
    
    finally:
        if owns:
            conn.close()

# function to delete a receipt (with associate items) from sb
def _sb_delete_receipt(receipt_id, supabase):
    supabase.table("receipt_items").delete().eq("receipt_id", receipt_id).execute()
    supabase.table("receipts").delete().eq("receipt_id", receipt_id).execute()

# function to delete a receipt (with associate items) from pg
def _pg_delete_receipt(receipt_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM receipt_items WHERE receipt_id = %s", (receipt_id,))
            cur.execute("DELETE FROM receipts WHERE receipt_id = %s", (receipt_id,))
        if owns:
            conn.commit()
    finally:
        if owns:
            conn.close()

# function to delete a receipt item from sb
def _sb_delete_item(item_id, supabase):
    supabase.table("receipt_items").delete().eq("id", item_id).execute()

# function to delete a receipt item from pg
def _pg_delete_item(item_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM receipt_items WHERE id = %s", (item_id,))
        if owns:
            conn.commit()
    finally:
        if owns:
            conn.close()

# function to fetch the receipt ID for a receipt item from sb
def _sb_get_receipt_id_for_item(item_id, supabase):
    row = (supabase.table("receipt_items").select("receipt_id").eq("id", item_id)
           .single().execute().data)
    return row["receipt_id"] if row else None

# function to fetch the receipt ID for a receipt item from pg
def _pg_get_receipt_id_for_item(item_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT receipt_id FROM receipt_items WHERE id = %s", (item_id,))
            r = cur.fetchone()
        return r["receipt_id"] if r else None
    finally:
        if owns:
            conn.close()

# function to fetch a receipt item from sb
def _sb_get_item(item_id, supabase):
    return (supabase.table("receipt_items").select("*").eq("id", item_id).single().execute().data)

# function to fetch a receipt item from pg
def _pg_get_item(item_id, conn=None):
    owns = conn is None
    if owns: conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM receipt_items WHERE id = %s", (item_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if owns: conn.close()

# function to update a receipt item in sb
def _sb_update_item(item_id, payload, supabase):
    supabase.table("receipt_items").update(payload).eq("id", item_id).execute()

# function to update a receipt item in pg
def _pg_update_item(item_id, payload, conn=None):
    if not payload: return
    owns = conn is None
    if owns: conn = get_conn()
    try:
        cols, vals = [], []
        for k, v in payload.items():
            cols.append(f"{k} = %s"); vals.append(v)
        vals.append(item_id)

        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE receipt_items SET {', '.join(cols)} WHERE id = %s",
                tuple(vals),
            )
        if owns: conn.commit()
    finally:
        if owns: conn.close()

# function to fetch items for a user from sb
def _sb_items_for_user(user_id, supabase):
    # fetch receipt_ids first, then items in it
    rids = (supabase.table("receipts").select("receipt_id").eq("user_id", user_id).execute().data) or []
    rids = [r["receipt_id"] for r in rids]
    if not rids: return []
    items = (supabase.table("receipt_items").select("*").in_("receipt_id", rids).execute().data) or []
    return items

# function to fetch items for a user from pg
def _pg_items_for_user(user_id, conn=None):
    owns = conn is None
    if owns:
        conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(""" 
              SELECT ri.* 
              FROM receipt_items ri 
              JOIN receipts r ON r.receipt_id = ri.receipt_id 
              WHERE r.user_id = %s 
            """, (user_id,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        if owns:
            conn.close()

# main public wrapper functions to call respective fucntion based on backend 
def get_companies(user_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_get_companies(user_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_get_companies(user_id, supabase)
    raise RuntimeError("Bad backend")

def get_receipts_page(user_id, filters, page, per_page, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_get_receipts_page(user_id, filters, page, per_page, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_get_receipts_page(user_id, filters, page, per_page, supabase)
    raise RuntimeError("Bad backend")

def get_receipt_for_user(receipt_id, user_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_get_receipt_for_user(receipt_id, user_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_get_receipt_for_user(receipt_id, user_id, supabase)
    raise RuntimeError("Bad backend")

def update_receipt(receipt_id, user_id, payload, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_update_receipt(receipt_id, user_id, payload, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_update_receipt(receipt_id, user_id, payload, supabase)
    raise RuntimeError("Bad backend")

def aggregate_totals(receipt_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_aggregate_totals(receipt_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_aggregate_totals(receipt_id, supabase)
    raise RuntimeError("Bad backend")

def delete_receipt(receipt_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_delete_receipt(receipt_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_delete_receipt(receipt_id, supabase)
    raise RuntimeError("Bad backend")

def delete_item(item_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_delete_item(item_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_delete_item(item_id, supabase)
    raise RuntimeError("Bad backend")

def get_receipt_id_for_item(item_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_get_receipt_id_for_item(item_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_get_receipt_id_for_item(item_id, supabase)
    raise RuntimeError("Bad backend")

def get_item(item_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_get_item(item_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_get_item(item_id, supabase)
    raise RuntimeError("Bad backend")

def update_item(item_id, payload, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_update_item(item_id, payload, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_update_item(item_id, payload, supabase)
    raise RuntimeError("Bad backend")

def items_for_user(user_id, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres": return _pg_items_for_user(user_id, conn)
    if DEFAULT_BACKEND == "supabase": return _sb_items_for_user(user_id, supabase)
    raise RuntimeError("Bad backend")

# -------------------------------------------
# dashboard route functions #

# function to fetch receipts and items for the dashboard with filters for sb
def _sb_fetch_rows(user_id, filters, supabase):
    # retrieve filters from the input (description, company values, date range)
    search_description = filters.get("description", "")
    company_values = filters.get("company_values") or []
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    # Narrow down the receipts based on company and date filters
    rq = (
        supabase.table("receipts")
        .select("receipt_id, purchase_date, company_name, taxes, upload_date")
        .eq("user_id", user_id)
    )
    if company_values:
        rq = rq.in_("company_name", company_values)
    if start_date:
        rq = rq.gte("purchase_date", start_date)
    if end_date:
        rq = rq.lte("purchase_date", end_date)

    receipts = rq.execute().data or []  # fetch receipts
    if not receipts:
        return [], [] 

    r_lookup = {r["receipt_id"]: r for r in receipts}  # create a lookup table for receipts
    rids = list(r_lookup.keys())  # list of receipt IDs to fetch associated items

    # fetch items for the selected receipts, including family_name field
    iq = supabase.table("receipt_items").select("*, family_name").in_("receipt_id", rids)
    if search_description:
        iq = iq.ilike("description", f"%{search_description}%")  # filter by description (case-insensitive)
    items = iq.execute().data or []  # fetch items

    # Attach receipt fields to each item
    for it in items:
        r = r_lookup.get(it["receipt_id"])
        if r:
            it["company_name"] = r.get("company_name")
            it["taxes"] = r.get("taxes")
            it["purchase_date"] = r.get("purchase_date")
            it["upload_date"] = r.get("upload_date")

    return items, [r for r in receipts]  # return the items and their associated receipts

# function to fetch rows for the dashboard from pg with filters
def _pg_fetch_rows_query(cur, user_id, description, company_values, start_date, end_date):
    # create the WHERE clause for the sql query based on the filters
    where = ["r.user_id = %s"]
    params = [user_id]
    
    # filter for description matching
    if description:
        where.append("ri.description ILIKE %s")  # filter by description (not case sensitive)
        params.append(f"%{description}%")
    # filter to match receipts from specific companies
    if company_values:
        placeholders = ",".join(["%s"] * len(company_values))  # create placeholders for company values
        where.append(f"r.company_name IN ({placeholders})")
        params.extend(company_values)
    # filter by receipts dates
    if start_date:
        where.append("r.purchase_date >= %s")
        params.append(start_date)
    if end_date:
        where.append("r.purchase_date <= %s")
        params.append(end_date)

    # Join all the filters to create the full WHERE clause
    where_sql = " AND ".join(where)
    # query to fetch items and associated receipt fields
    cur.execute(
        f"""
        SELECT
          ri.*,
          r.company_name,
          r.taxes,
          r.purchase_date,
          r.upload_date,
          ri.family_name
        FROM receipt_items ri
        JOIN receipts r ON r.receipt_id = ri.receipt_id
        WHERE {where_sql}
        """,
        tuple(params),
    )
    return [dict(r) for r in cur.fetchall()]  

# function to fetch the list of unique company names from pg
def _pg_company_list_query(cur, user_id):
    cur.execute(
        """
        SELECT DISTINCT company_name
        FROM receipts
        WHERE user_id = %s AND company_name IS NOT NULL
        ORDER BY company_name
        """,
        (user_id,),
    )
    rows = cur.fetchall()  # fetch the company names
    return [r["company_name"] for r in rows]  # return the list of company names

# function to fetch rows for the dashboard with company list from pg
def _pg_fetch_rows(user_id, filters, conn=None):
    owns_conn = conn is None  
    if owns_conn:
        conn = get_conn()

    search_description = (filters.get("description") or "").strip()
    company_values = filters.get("company_values") or []
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    try:
        if owns_conn:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    items = _pg_fetch_rows_query(
                        cur, user_id, search_description, company_values, start_date, end_date
                    )
                    companies = _pg_company_list_query(cur, user_id)
                    # companies = _pg_get_companies(cur, user_id)
        else:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                items = _pg_fetch_rows_query(
                    cur, user_id, search_description, company_values, start_date, end_date
                )
                # companies = _pg_company_list_query(cur, user_id)
                companies = _pg_get_companies(cur, user_id)

        return items, companies
    finally:
        if owns_conn:
            conn.close()

# function to get rows for table/visuals in dashboard
def fetch_dashboard_rows(user_id, filters, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres":
        # Returns (items, companies) already, including family_name
        # companies is for drop down filter in dashboard
        items, companies = _pg_fetch_rows(user_id, filters, conn=conn)
        return items, companies
    elif DEFAULT_BACKEND == "supabase":
        items, _ = _sb_fetch_rows(user_id, filters, supabase=supabase)
        # companies = _sb_company_options(user_id, supabase=supabase)
        companies =  _sb_get_companies(user_id, supabase=supabase)
        return items, companies
    else:
        raise RuntimeError(f"Unknown backend: {DEFAULT_BACKEND}")
    
# function to fetch export rows for a user from sb (for exporting data)
def _sb_fetch_export_rows(user_id, filters, supabase):
    # Extract filters from the 'filters' dictionary with default values for each filter
    description = (filters.get("description") or "").strip()
    company_values = filters.get("company_values") or []
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    # get only the user's receipts (apply company/date filters)
    rq = (supabase.table("receipts")
          .select("receipt_id, company_name, purchase_date")
          .eq("user_id", user_id))
    if company_values:
        rq = rq.in_("company_name", company_values)
    if start_date:
        rq = rq.gte("purchase_date", start_date)
    if end_date:
        rq = rq.lte("purchase_date", end_date)

    receipts = rq.execute().data or []  
    if not receipts:
        return []  

    rmap = {r["receipt_id"]: r for r in receipts}  # map receipt_id to receipt data

    # Fetch items for these receipt IDs (optional description filter)
    iq = supabase.table("receipt_items").select("*").in_("receipt_id", list(rmap.keys()))
    if description:
        iq = iq.ilike("description", f"%{description}%")  # filter items by description (case-insensitive)
    items = iq.execute().data or []  # fetch items, or return empty list if none found

    # Flatten the data by attaching receipt fields to each item
    for it in items:
        rec = rmap.get(it["receipt_id"], {})
        it["purchase_date"] = rec.get("purchase_date")
        it["company_name"] = rec.get("company_name")
    return items  # return the list of items with receipt fields attached

# function for fetching export rows for a user from pg (for exporting data)
def _pg_fetch_export_rows(user_id, filters, conn=None):
    owns_conn = conn is None
    if owns_conn:
        conn = get_conn()  

    description    = (filters.get("description") or "").strip()
    company_values = filters.get("company_values") or []
    start_date     = filters.get("start_date")
    end_date       = filters.get("end_date")

    try:
        where = ["r.user_id = %s"]
        params = [user_id]

        if description:
            where.append("ri.description ILIKE %s")
            params.append(f"%{description}%")

        if company_values:
            where.append("r.company_name = ANY(%s)")
            params.append(company_values)

        if start_date:
            where.append("r.purchase_date >= %s")
            params.append(start_date)

        if end_date:
            where.append("r.purchase_date <= %s")
            params.append(end_date)

        where_sql = " AND ".join(where)

        query = f"""
            SELECT
              ri.*,
              r.purchase_date,
              r.company_name
            FROM receipt_items ri
            JOIN receipts r ON r.receipt_id = ri.receipt_id
            WHERE {where_sql}
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        return [dict(row) for row in rows]

    finally:
        if owns_conn:
            conn.close()

# public function to get export rows, return with (all receipt_items cols, purchase_date and company name with filters)
def fetch_export_rows(user_id, filters, *, supabase=None, conn=None):
    if DEFAULT_BACKEND == "postgres":
        return _pg_fetch_export_rows(user_id, filters, conn=conn)
    elif DEFAULT_BACKEND == "supabase":
        if supabase is None:
            raise RuntimeError("Supabase client not provided.")
        return _sb_fetch_export_rows(user_id, filters, supabase=supabase)
    else:
        raise RuntimeError(f"Unknown backend: {DEFAULT_BACKEND}")

#----------------------------------------------
# Main APP ROUTES WITH HELPER FUNCTIONS ON TOP

# start flask app
app = Flask(__name__)
app.secret_key = "secret"  

# start local postgresql db is not created
init_db()

# route for home (if not logged in, bring to log in page, else bring to dashboard page)
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('receipts'))

# route for register user
@app.route('/register', methods=['GET', 'POST'])
def register():
    # check if they are logged in if yes, bring to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    # get input for username and passwrod
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        # hash password
        password_hash = generate_password_hash(password)
        # inserting data into users table based on backend 
        if DEFAULT_BACKEND == "supabase":
            user, err = create_user(username, password_hash, supabase=supabase)
        else:
            # postgres
            with get_conn() as conn:
                user, err = create_user(username, password_hash, conn=conn)
        if err == "exists":
            flash("Username already exists.")
            return render_template('register.html')

        flash("Registered successfully. Please login.")
        # bring user to login page (re-key to enter the web app)
        return redirect(url_for('login'))
    return render_template('register.html')

# login page route for user to login and access their data
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))  
    
    # If the form is submitted (POST request)
    if request.method == 'POST':
        username = request.form['username'].strip()  
        password = request.form['password']

        if DEFAULT_BACKEND == "supabase":
            user = get_user_by_username(username, supabase=supabase)  # Get user from sb
        else:
            # get user from pg
            with get_conn() as conn:
                user = get_user_by_username(username, conn=conn)

        # Verify if the user exists and the password matches (using hashed password comparison)
        if user and check_password_hash(user["password_hash"], password):
            session['user_id'] = user["id"]  # Store user_id in session
            session['username'] = user["username"]  # Store username in session
            return redirect(url_for('dashboard'))  
        else:
            flash("Invalid credentials.") 

    return render_template('login.html') 

# Route for logout
@app.route('/logout')
def logout():
    session.clear()  # Clear session data to log out the user
    flash("You have been logged out.", "info") 
    return redirect(url_for('login'))  # bring to login page after logout


ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png'}
def is_allowed_file(filename):
    return os.path.splitext(filename)[-1].lower() in ALLOWED_EXTENSIONS

@app.route('/receipts', methods=['GET', 'POST'])
@login_required
def receipts():
    # if 'user_id' not in session:
    #     flash("Please log in to add receipts.", "warning")
    #     return redirect(url_for('login'))
    if request.method == 'POST':
        # 3 ways of uploading
        form_type = request.form.get('form_type')

        if form_type == 'manual_entry':
            company_name = request.form.get('company_name')
            purchase_date = request.form.get('purchase_date')
            taxes = float(request.form.get('taxes') or 0.0)

            items = []
            descriptions = request.form.getlist('description[]')
            quantities = request.form.getlist('quantity[]')
            unit_prices = request.form.getlist('unit_price[]')
            discounts = request.form.getlist('discount[]')

            total_before_tax = total_discount = total_after_tax = 0.0
            # Loop through each item and calculate prices
            for i in range(len(descriptions)):
                try:
                    desc = descriptions[i]
                    qty = float(quantities[i])
                    unit = float(unit_prices[i])
                    disc = discounts[i]

                    # Skip invalid or zero quantity/price items
                    if not desc or qty <= 0 or unit <= 0:
                        continue

                    # Calculate original price, discount amount, and discounted price
                    original_price = round(qty * unit, 2)
                    discount_amt = compute_discount(disc, qty, unit, source="manual")
                    discounted_price = round(original_price - discount_amt, 2)
                    unit_price_after_discount = round(discounted_price / qty, 2)

                    # Add item details to the items list
                    item = {
                        "description": desc,
                        "quantity": qty,
                        "unit_price": unit,
                        "discount": disc,
                        "original_price": original_price,
                        "discounted_price": discounted_price,
                        "unit_price_after_discount": unit_price_after_discount,
                        "total_price": discounted_price
                    }

                    items.append(item)  # Append item to list
                    total_before_tax += original_price
                    total_discount += discount_amt
                    total_after_tax += discounted_price
                except Exception:
                    continue  

            # If there are valid items, save the receipt
            if items:
                receipt_data = {
                    "company_name": company_name,
                    "purchase_date": purchase_date,
                    "total_before_tax": round(total_before_tax, 2),
                    "taxes": round(taxes, 2),
                    "total_after_tax": round(total_after_tax + taxes, 2),
                    "total_discount": round(total_discount, 2),
                    "items": items,
                    "source": "manual"
                }
                save_receipt_for_current_user(receipt_data)  # Save the receipt to the database
                flash(f"Successfully added receipt with {len(items)} item(s).", "success")  
            else:
                flash("No valid items submitted.", "warning") 

        elif form_type == 'upload':
            files = request.files.getlist("receipt_files")  # Get list of uploaded files
            uploaded = 0  # count successful uploads

            # Create a temporary directory to process the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in files:
                    if file and file.filename:
                        original_filename = secure_filename(file.filename)  # Secure the filename
                        ext = os.path.splitext(original_filename)[-1].lower()  # Get file extension
                        if not is_allowed_file(original_filename):  # Check if the file is allowed
                            flash(f"unsupported file type: {original_filename}", "warning")
                            continue  

                        # Generate unique filename and save file temporarily
                        unique_filename = f"{uuid.uuid4()}{ext}"
                        temp_path = os.path.join(temp_dir, unique_filename)

                        try:
                            file.save(temp_path)  # Save the file to temporary directory
                            receipts = extract_receipt_data(temp_path)  # Extract receipt data from file
                            for receipt in receipts:
                                # convert date to purchase_date
                                if "date" in receipt and "purchase_date" not in receipt:
                                    receipt["purchase_date"] = (receipt.pop("date") or "").strip()
                                # change these to float
                                for k in ("total_before_tax", "taxes", "total_after_tax", "total_discount"):
                                    try:
                                        receipt[k] = float(str(receipt.get(k) or 0).replace(",", ""))
                                    except Exception:
                                        receipt[k] = 0.0
                                # create dictionary receipt with an empty list ([]) as the default value if it does not have the key
                                receipt.setdefault("items", [])
                                receipt.setdefault("source", "upload")

                                save_receipt_for_current_user(receipt)  # Save the extracted receipt data
                                uploaded += 1  
                        except Exception as e:
                            flash(f"Failed to process {original_filename}: {e}", "error")
                            continue  

            if uploaded > 0:
                flash(f"Processed and saved {uploaded} receipt(s).", "success")
            else:
                flash("No valid receipts were uploaded.", "warning")

        elif form_type == 'camera_upload':
            image_file = request.files.get('image')  # Get the image file from form input
            if image_file and image_file.filename:
                original_filename = secure_filename(image_file.filename)  # Secure the filename
                if not is_allowed_file(original_filename):
                    return jsonify({"status": "error", "message": "Unsupported image format."}), 400

                ext = os.path.splitext(original_filename)[-1].lower()
                uploaded = 0

                with tempfile.TemporaryDirectory() as temp_dir:
                    unique_filename = f"{uuid.uuid4()}{ext}"
                    temp_path = os.path.join(temp_dir, unique_filename)

                    try:
                        image_file.save(temp_path)  # Save to temp dir

                        receipts = extract_receipt_data(temp_path)
                        for receipt in receipts:
                            if "date" in receipt and "purchase_date" not in receipt:
                                receipt["purchase_date"] = (receipt.pop("date") or "").strip()
                            for k in ("total_before_tax", "taxes", "total_after_tax", "total_discount"):
                                try:
                                    receipt[k] = float(str(receipt.get(k) or 0).replace(",", ""))
                                except Exception:
                                    receipt[k] = 0.0
                            receipt.setdefault("items", [])
                            receipt.setdefault("source", "camera_upload")

                            save_receipt_for_current_user(receipt)
                            uploaded += 1

                        return jsonify({
                            "status": "success",
                            "message": f"Processed and saved {uploaded} receipt(s)."
                        }), 200

                    except Exception as e:
                        return jsonify({
                            "status": "error",
                            "message": f"Error processing image: {str(e)}"
                        }), 500
            else:
                return jsonify({"status": "error", "message": "No image file received."}), 400
        return redirect(url_for('receipts'))  # Redirect to the receipts page after form submission
    return render_template('receipts.html')  # Render the receipts page for GET request (fetch html)

# route to view receipts through show/hide table 
@app.route('/receipts/view')
@login_required
def view_receipts():
    user_id = session['user_id']

    if DEFAULT_BACKEND == "supabase":
        receipts, items_by_receipt = fetch_receipts_and_items(user_id, supabase=supabase)
    else:
        with get_conn() as conn:
            receipts, items_by_receipt = fetch_receipts_and_items(user_id, conn=conn)

    # Debug logging
    print(f"[DEBUG] User {user_id} - Receipts found: {len(receipts)}")
    print(f"[DEBUG] User {user_id} - Items found: {sum(len(v) for v in items_by_receipt.values())}")
    return render_template("view_receipts.html",
                           receipts=receipts,
                           items_by_receipt=items_by_receipt)

# route for receipts_table (managing receipts and items)
@app.route('/receipts_table', methods=['GET', 'POST'])
@login_required
def receipts_table():
    # defines how many receipts shown per page
    PER_PAGE = 10
    page = int(request.args.get("page", 1))   # get current page number from query param (default = 1)
    user_id = session['user_id']             

    # build filters dictionary (support both POST form submissions and GET query params)
    filters = {
        "company_name": request.form.get("company_name", "").strip() if request.method == "POST" else (request.args.get("company_name", "") or "").strip(),
        "purchase_date": request.form.get("purchase_date", "").strip() if request.method == "POST" else (request.args.get("purchase_date", "") or "").strip()
    }

    # query receipts and companies depending on backend 
    if DEFAULT_BACKEND == "supabase":
        companies = get_companies(user_id, supabase=supabase)  
        receipts, items_by_receipt, total_pages = get_receipts_page(user_id, filters, page, PER_PAGE, supabase=supabase)
    else:
        with get_conn() as conn:
            companies = get_companies(user_id, conn=conn)
            receipts, items_by_receipt, total_pages = get_receipts_page(user_id, filters, page, PER_PAGE, conn=conn)

    # render receipts table template with pagination and filters (with receipts and items)
    return render_template(
        "receipts_table.html",
        receipts=receipts,
        filters=filters,
        companies=companies,
        items_by_receipt=items_by_receipt,
        page=page,
        total_pages=total_pages
    )


# edit_receipt route to update receipt metadata (company, date, taxes, manual total)
@app.route('/edit_receipt/<receipt_id>', methods=['POST'])
@login_required
def edit_receipt(receipt_id):
    user_id = session['user_id']  

    # get form values safely (strip spaces and handle empty input)
    company_name    = (request.form.get('company_name') or '').strip()
    purchase_date_s = (request.form.get('purchase_date') or '').strip()
    taxes_s         = (request.form.get('taxes') or '').strip()
    manual_flag     = bool(request.form.get('enable_manual_total'))  # checkbox for manual total override
    total_after_s   = (request.form.get('total_after_tax') or '').strip()

    #company name must not be empty
    if not company_name:
        flash("Company name cannot be empty.", "warning")
        return redirect(url_for('receipts_table'))

    #check purchase date format (YYYY-MM-DD)
    if purchase_date_s:
        try:
            datetime.strptime(purchase_date_s, "%Y-%m-%d")
        except ValueError:
            flash("Purchase date must be YYYY-MM-DD.", "warning")
            return redirect(url_for('receipts_table'))

    # taxes must be positive number
    try:
        taxes_val = float(taxes_s.replace(',', '').replace('$', '')) if taxes_s else None
        if taxes_val is not None and taxes_val < 0:
            raise ValueError()
    except ValueError:
        flash("Taxes must be a non-negative number.", "warning")
        return redirect(url_for('receipts_table'))

    # handle manual override of total_after_tax
    tat_new = None
    if manual_flag and total_after_s:
        try:
            tat_new = float(total_after_s.replace(',', '').replace('$', ''))
            if tat_new < 0:
                raise ValueError()
        except ValueError:
            flash("Manual Total After Tax must be a non-negative number.", "warning")
            return redirect(url_for('receipts_table'))

    # ensure the receipt belongs to the current user
    if DEFAULT_BACKEND == "supabase":
        rec = get_receipt_for_user(receipt_id, user_id, supabase=supabase)
    else:
        conn = get_conn()
        try:
            rec = get_receipt_for_user(receipt_id, user_id, conn=conn)
        finally:
            conn.close()

    if not rec:
        flash("Receipt not found or not yours.", "danger")
        return redirect(url_for('receipts_table'))

    # build payload for updating receipt
    update_payload = {"company_name": company_name}
    if purchase_date_s:
        update_payload["purchase_date"] = purchase_date_s
    if taxes_val is not None:
        update_payload["taxes"] = taxes_val

    # handle manual override flag and total
    if manual_flag:
        update_payload["total_after_tax_override"] = True
        if tat_new is not None:
            update_payload["total_after_tax"] = tat_new
        respect_override = True  # tells recalc to respect manual total
    else:
        update_payload["total_after_tax_override"] = False
        update_payload["total_after_tax"] = None
        respect_override = False

    # persist update to backend and recalculate totals
    if DEFAULT_BACKEND == "supabase":
        update_receipt(receipt_id, user_id, update_payload, supabase=supabase)
        recalculate_receipt_totals(receipt_id, respect_override=respect_override, supabase=supabase)
    else:
        conn = get_conn()
        try:
            update_receipt(receipt_id, user_id, update_payload, conn=conn)
            recalculate_receipt_totals(receipt_id, respect_override=respect_override, conn=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    flash("Receipt updated.", "success")
    return redirect(url_for('receipts_table'))

# api_receipt_totals route (returns receipt totals as JSON (for AJAX/JS))
@app.route("/api/receipt_totals/<receipt_id>")
@login_required
def api_receipt_totals(receipt_id):
    # fetch aggregated totals for a receipt 
    if DEFAULT_BACKEND == "supabase":
        agg = aggregate_totals(receipt_id, supabase=supabase)
    else:
        with get_conn() as conn:
            agg = aggregate_totals(receipt_id, conn=conn)

    if not agg:
        return jsonify({"error": "Receipt not found"}), 404

    # return values formatted to 2 decimals
    return jsonify({
        "subtotal": f"{agg['subtotal']:.2f}",
        "discount_total": f"{agg['discount_total']:.2f}",
        "taxes": f"{agg['taxes']:.2f}",
        "total_after_tax": f"{agg['total_after_tax']:.2f}",
    })


# delete_receipt route (deletes an entire receipt and its items)
@app.route('/delete_receipt/<receipt_id>', methods=['POST'])
@login_required
def delete_receipt_route(receipt_id):
    try:
        #use helper function to delete receipt
        if DEFAULT_BACKEND == "supabase":
            delete_receipt(receipt_id, supabase=supabase)
        else:
            conn = get_conn()
            try:
                delete_receipt(receipt_id, conn=conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        flash("Receipt deleted.", "success")
    except Exception as e:
        app.logger.exception(f"Failed to delete receipt {receipt_id}")
        flash(f"Error deleting receipt: {e}", "danger")
    return redirect(url_for('receipts_table', status='receipt_deleted'))


# delete_item route (deletes a single item from a receipt and recalculates totals)
@app.route('/delete_item/<int:item_id>/<receipt_id>', methods=['POST'])
@login_required
def delete_item_route(item_id, receipt_id):
    try:
        # use helper functions to delete and recalculate
        if DEFAULT_BACKEND == "supabase":
            delete_item(item_id, supabase=supabase)
            recalculate_receipt_totals(receipt_id, supabase=supabase)
        else:
            conn = get_conn()
            try:
                delete_item(item_id, conn=conn)
                recalculate_receipt_totals(receipt_id, conn=conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        flash("Item deleted.", "success")
    except Exception as e:
        app.logger.exception(f"Failed to delete item {item_id} from receipt {receipt_id}")
        flash(f"Error deleting item: {e}", "danger")
    return redirect(url_for('receipts_table', status='item_deleted'))


# delete_multiple route (deletes multiple receipts and/or items at a time)
@app.route('/delete_multiple', methods=['POST'])
@login_required
def delete_multiple():
    # get selected receipts and items from form
    receipt_ids = request.form.getlist("selected_receipts")
    item_ids = request.form.getlist("selected_items")
    try:
        if DEFAULT_BACKEND == "supabase":
            # delete receipts
            for rid in receipt_ids:
                delete_receipt(rid, supabase=supabase)
            # delete items and recalc totals
            for iid in item_ids:
                rid = get_receipt_id_for_item(iid, supabase=supabase)
                if rid:
                    delete_item(iid, supabase=supabase)
                    recalculate_receipt_totals(rid, supabase=supabase)
        else:
            conn = get_conn()
            try:
                for rid in receipt_ids:
                    delete_receipt(rid, conn=conn)
                for iid in item_ids:
                    rid = get_receipt_id_for_item(iid, conn=conn)
                    if rid:
                        delete_item(iid, conn=conn)
                        recalculate_receipt_totals(rid, conn=conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        flash("Deleted selected receipts/items.", "success")
    except Exception as e:
        app.logger.exception("Bulk delete failed")
        flash(f"Error deleting: {str(e)}", "danger")
    return redirect(url_for('receipts_table', status='bulk_deleted'))


# edit_item route (GET/POST) (edit a single item inside a receipt new page)
@app.route('/edit_item/<int:item_id>', methods=['GET', 'POST'])
@login_required
def edit_item_route(item_id):
    if request.method == "POST":
        try:
            # collect input from users
            description = (request.form["description"] or "").strip()
            quantity = float(request.form["quantity"] or 0)
            unit_price = float(request.form["unit_price"] or 0)
            discount = (request.form.get("discount") or "").strip()

            # validation: must have description and positive numbers
            if not description or quantity <= 0 or unit_price <= 0:
                flash("Invalid data.", "warning")
                return redirect(url_for("edit_item_route", item_id=item_id))

            # find receipt id for this item
            if DEFAULT_BACKEND == "supabase":
                rid = get_receipt_id_for_item(item_id, supabase=supabase)
            else:
                conn = get_conn()
                try:
                    rid = get_receipt_id_for_item(item_id, conn=conn)
                finally:
                    conn.close()

            if not rid:
                flash("Item not found.", "warning")
                return redirect(url_for("receipts_table"))

            # calculate price values
            original_price = round(quantity * unit_price, 2)
            discount_amt = compute_discount(discount, quantity, unit_price, source="edit") or 0.0
            discounted_price = round(original_price - discount_amt, 2)
            unit_price_after_discount = round(discounted_price / quantity, 2)

            # prepare update payload
            payload = {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "original_price": original_price,
                "discounted_price": discounted_price,
                "unit_price_after_discount": unit_price_after_discount,
                "total_price": discounted_price
            }

            # update item and recalc totals using helper functions
            if DEFAULT_BACKEND == "supabase":
                update_item(item_id, payload, supabase=supabase)
                recalculate_receipt_totals(rid, supabase=supabase)
            else:
                conn = get_conn()
                try:
                    update_item(item_id, payload, conn=conn)
                    recalculate_receipt_totals(rid, conn=conn)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            flash("Item updated successfully.", "success")
            return redirect(url_for("receipts_table", status="edit_success"))

        except Exception as e:
            app.logger.exception(f"Error editing item {item_id}")
            flash(f"Error: {e}", "danger")
            return redirect(url_for("edit_item_route", item_id=item_id))

    # else GET request (fetch item details to show in form)
    if DEFAULT_BACKEND == "supabase":
        item = get_item(item_id, supabase=supabase)
    else:
        conn = get_conn()
        try:
            item = get_item(item_id, conn=conn)
        finally:
            conn.close()

    if not item:
        flash("Item not found.", "warning")
        return redirect(url_for("receipts_table"))
    return render_template("edit_item.html", item=item)


# export_receipts_csv route (exports all items into CSV)
@app.route('/export_receipts_csv')
@login_required
def export_receipts_csv():
    user_id = session["user_id"]

    if DEFAULT_BACKEND == "supabase":
        items = items_for_user(user_id, supabase=supabase)
    else:
        with get_conn() as conn:
            items = items_for_user(user_id, conn=conn)

    # convert to dataframe then CSV
    df = pd.DataFrame(items)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    # return downloadable CSV response
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=receipt_items.csv"})

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    user_id = session['user_id']
    # Collect user-provided filters 
    search_description = request.args.get('description', '').strip()
    company_values = request.args.getlist('company')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # make filters into a dictionary for passing into fetch functions
    filters = {
        "description": search_description,
        "company_values": company_values,
        "start_date": start_date,
        "end_date": end_date,
    }

    print("Dashboard accessed by user_id:", user_id)

    # fetch rows via toggle-aware store
    if DEFAULT_BACKEND == "supabase":
        rows, company_options_list = fetch_dashboard_rows(user_id, filters, supabase=supabase)
    else:
        conn = get_conn()
        try:
            rows, company_options_list = fetch_dashboard_rows(user_id, filters)
        finally:
            conn.close()

    # Convert to DataFrame 
    df = pd.DataFrame(rows)
    if not df.empty:
        # Convert numerical/text/date fields to appropriate types
        df['unit_price'] = pd.to_numeric(df.get('unit_price'), errors='coerce')
        df['taxes'] = pd.to_numeric(df.get('taxes'), errors='coerce')
        df['purchase_date'] = pd.to_datetime(df.get('purchase_date'), errors='coerce')

        # Add the 'family_name' column for display
        if 'family_name' in df.columns:
            df['family_name'] = df['family_name'].fillna('Unknown')  

        # Apply user input filters on the DataFrame
        if search_description:
            df = df[df['description'].astype(str).str.contains(search_description, case=False, na=False)]
        if company_values:
            df = df[df['company_name'].isin(company_values)]
        if start_date:
            df = df[df['purchase_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['purchase_date'] <= pd.to_datetime(end_date)]

        # Clean version of DataFrame for displaying in HTML table (drop unused columns)
        df_for_table = df.drop(columns=['month'], errors='ignore')
    else:
        # If no rows, create empty DataFrame placeholder
        df_for_table = pd.DataFrame()

    # Company dropdown values 
    company_options_var = company_options_list 

    # Summary and charts 
    summary = {}
    charts = {}

    if not df.empty:
        #  consistent datas types 
        for col in ["quantity", "unit_price", "unit_price_after_discount", "total_price", "taxes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col],  errors="coerce")

        if "purchase_date" in df.columns:
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")

        # Derive 'month' if missing (YYYY-MM)
        if "month" not in df.columns:
            if "purchase_date" in df.columns:
                df["month"] = df["purchase_date"].dt.to_period("M").astype(str)
            else:
                # when no purchase_date is available
                df["month"] = "Unknown"

        # Summary Stats features

        # Most common item by total quantity purchased
        # check if columns presense and quantity not null
        if "description" in df.columns and "quantity" in df.columns and df["quantity"].notna().any():
            qty_by_item = df.groupby("description", dropna=False)["quantity"].sum().sort_values(ascending=False)
            summary["most_common"] = qty_by_item.index[0] if not qty_by_item.empty else None
        else:
            summary["most_common"] = None

        # Highest tax seen on any row
        summary["highest_tax"] = float(df["taxes"].max()) if "taxes" in df.columns and df["taxes"].notna().any() else None

        # Most expensive single unit (by unit_price)
        if "unit_price" in df.columns and df["unit_price"].notna().any():
            most_expensive = df.loc[df["unit_price"].idxmax()]
            summary["most_expensive"] = {
                "desc": most_expensive.get("description", ""),
                "price": float(most_expensive["unit_price"])
            }
        else:
            summary["most_expensive"] = {"desc": None, "price": None}

        # Get the grouping preference from the request (either description or family_name)
        group_by = request.args.get('group_by', 'description')

        # Group by either family_name or description based on the user's selection
        if group_by == 'family_name':
            group_by_col = 'family_name'
        else:
            group_by_col = 'description'

        # Chart Data
        # Create "year_month" column from YYYY-MM-DD for time-series analysis
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
        df["year_month"] = df["purchase_date"].dt.to_period("M").astype(str)

        # 1) Monthly Spending Trend chart
        if "total_price" in df.columns and df["total_price"].notna().any():
            monthly_spending = (
                df.dropna(subset=["year_month", "total_price"])
                .groupby("year_month", as_index=False)["total_price"].sum()
            )

            # Fill in missing months with zero values to ensure continuity
            all_months = pd.period_range(
                df["purchase_date"].min().to_period("M"),
                df["purchase_date"].max().to_period("M"),
                freq="M"
            ).astype(str)

            monthly_spending = (
                monthly_spending.set_index("year_month")
                .reindex(all_months, fill_value=0)
                .reset_index()
                .rename(columns={"index": "year_month"})
            )

            # Construct Plotly line chart
            line_chart = go.Figure()
            line_chart.add_trace(
                go.Scatter(
                    x=monthly_spending["year_month"],
                    y=monthly_spending["total_price"],
                    mode="lines+markers",
                    name="Monthly Total Spending (Excluding Tax)"
                )
            )
            line_chart.update_layout(
                xaxis_title="Month",
                yaxis_title="Total Spending",
                title="Monthly Spending Total",
                template="plotly_white",
                showlegend=True
            )
        else:
            line_chart = go.Figure()

        # 2) Top 5 spending categories/items (Pie chart)
        if group_by_col in df.columns and "total_price" in df.columns and df["total_price"].notna().any():
            pie_df = (
                df.dropna(subset=[group_by_col, "total_price"])
                .groupby(group_by_col)["total_price"].sum()
                .nlargest(5)
            )
            pie_chart = go.Figure(data=[go.Pie(labels=pie_df.index, values=pie_df.values, hole=0)])
        else:
            pie_chart = go.Figure()

        # 3) Top 5 frequently bought items (Bar chart by quantity)
        if "quantity" in df.columns and df["quantity"].notna().any():
            bar_df = (
                df.dropna(subset=[group_by_col, "quantity"])
                .groupby(group_by_col, as_index=False)["quantity"].sum()
                .sort_values("quantity", ascending=False)
                .head(5)
            )
            bar_chart = go.Figure(data=[go.Bar(x=bar_df[group_by_col], y=bar_df["quantity"], name="Quantity")])
        else:
            bar_chart = go.Figure()

        # Save chart json payloads for rendering in Jinja2
        charts.update({
            "line_chart_data": json.dumps(line_chart["data"], cls=PlotlyJSONEncoder),
            "line_chart_layout": json.dumps(line_chart["layout"], cls=PlotlyJSONEncoder),
            "pie_chart_data": json.dumps(pie_chart["data"], cls=PlotlyJSONEncoder),
            "pie_chart_layout": json.dumps(pie_chart["layout"], cls=PlotlyJSONEncoder),
            "bar_chart_data": json.dumps(bar_chart["data"], cls=PlotlyJSONEncoder),
            "bar_chart_layout": json.dumps(bar_chart["layout"], cls=PlotlyJSONEncoder),
        })

        # 4) Unit price trend for searched item(s) (line chart over time)
        if search_description and "description" in df.columns:
            mask = df["description"].astype(str).str.contains(search_description, case=False, na=False)
            search_df = df.loc[mask].copy()

            if not search_df.empty and "purchase_date" in search_df.columns and "unit_price_after_discount" in search_df.columns:
                search_df = search_df.sort_values("purchase_date")
                price_trend = go.Figure()
                cheapest_info = []  # Track cheapest observed unit price

                for item_name, item_df in search_df.groupby("description", dropna=False):
                    item_df = item_df.dropna(subset=["purchase_date", "unit_price_after_discount"])
                    if item_df.empty:
                        continue

                    # Add a trace for each distinct item
                    price_trend.add_trace(go.Scatter(
                        x=item_df["purchase_date"],
                        y=item_df["unit_price_after_discount"],
                        mode="lines+markers",
                        name=str(item_name)
                    ))

                    # Identify the cheapest observed unit price for this item
                    min_idx = item_df["unit_price_after_discount"].idxmin()
                    min_row = item_df.loc[min_idx]
                    cheapest_info.append({
                        "description": str(item_name),
                        "price": round(float(min_row["unit_price_after_discount"]), 2),
                        "date": min_row["purchase_date"].strftime("%Y-%m-%d") if pd.notna(min_row["purchase_date"]) else None
                    })

                # Save price trend chart + cheapest item info
                charts["price_trend_data"] = json.dumps(price_trend["data"], cls=PlotlyJSONEncoder)
                charts["price_trend_layout"] = json.dumps(price_trend["layout"], cls=PlotlyJSONEncoder)
                charts["cheapest_info"] = cheapest_info

    # Render the dashboard template with processed data, summaries, and charts
    return render_template(
        'dashboard.html',
        df=df,
        df_for_table=df_for_table,
        summary=summary,
        charts=charts,
        company_options=company_options_var,
        selected_companies=company_values
    )

@app.route('/dashboard/download')
@login_required
def download_csv():
    user_id = session['user_id']
    # get user input filter
    description    = request.args.get('description', '').strip()
    start_date     = request.args.get('start_date')
    end_date       = request.args.get('end_date')
    company_values = request.args.getlist('company')

    # create a dict with filters
    filters = {
        "description": description,
        "company_values": company_values,
        "start_date": start_date,
        "end_date": end_date,
    }

    # sends in filters for getting only the required data
    if DEFAULT_BACKEND == "supabase":
        rows = fetch_export_rows(user_id, filters, supabase=supabase)
    else:
        with get_conn() as conn:
            rows = fetch_export_rows(user_id, filters, conn=conn)

    if not rows:
        flash("No data found to export.", "warning")
        return redirect(url_for('dashboard'))

    # convert to pandas
    df = pd.DataFrame(rows)

    # Export to Excel
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="filtered_receipts.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

#run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
