1. Replace the Api_key and Supabase_key with actual the actual API keys

2. Select the backend based on requirements.
"supabase" connects to a cloud database hosted on Supabase.
"postgres" connects to a local PostgreSQL database.

3. The prompt for the Gemini 2.5 Pro/Flash multimodal model is designed to ignore plastic bag–related items and items without a price (e.g., free gifts), as these are assumed not to being purchased by the user.

4. Adjust the cosine similarity threshold if needed.
If the user wants to treat "wang lao ji 500ml" and "(500ml) of wang lao ji" as the same item, they should set a higher threshold (around 0.8).
If they want to be more lenient (e.g., treating "wang lao ji" as the same group as "wang lao ji 500ml"), they should use a lower threshold. However, keep in mind that "wang lao ji" without a size might actually refer to a different product variant (e.g., 1L vs. 500ml), so lowering the threshold may group distinct products together.
The default threshold is 0.82.
