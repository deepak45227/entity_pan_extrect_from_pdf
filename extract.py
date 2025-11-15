import os
import csv
import json
import time
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts text from all pages of a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None
        
    print(f"Reading text from '{pdf_path}'...")
    try:
        reader = PdfReader(pdf_path)
        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        
        if not text_parts:
            print("Warning: No text could be extracted from PDF. It might be image-based.")
            return None
            
        full_text = "\n".join(text_parts)
        print(f"Successfully extracted {len(full_text)} characters from {len(text_parts)} pages.")
        return full_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def get_available_models(api_key: str) -> list[str]:
    """Returns list of available models sorted by preference."""
    try:
        genai.configure(api_key=api_key)
        
        # Preferred models for free tier (flash models have higher quotas)
        preferred_order = [
            "gemini-2.0-flash",           # Free tier friendly
            "gemini-2.5-flash",           # Free tier friendly
            "gemini-flash-latest",        # Fallback
            "gemini-2.0-flash-lite",      # More quota
            "gemini-2.5-flash-lite",      # More quota
            "gemini-2.5-pro",             # Powerful but limited quota
            "gemini-pro-latest",          # Fallback
        ]
        
        available_models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        # Sort available models by preference
        sorted_models = []
        for preferred in preferred_order:
            for available in available_models:
                if preferred in available and available not in sorted_models:
                    sorted_models.append(available)
        
        # Add any remaining models
        for available in available_models:
            if available not in sorted_models:
                sorted_models.append(available)
        
        return sorted_models
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def chunk_text(text: str, max_chars: int = 120000) -> list[str]:
    """Splits text into chunks to avoid token limits."""
    chunks = []
    current_chunk = ""
    
    # Split by pages or paragraphs
    sections = text.split('\n\n')
    
    for section in sections:
        if len(current_chunk) + len(section) < max_chars:
            current_chunk += section + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_entities_from_chunk(model, chunk: str, chunk_num: int, total_chunks: int, 
                                max_retries: int = 3) -> list[dict]:
    """Extract entities from a single chunk of text with retry logic."""
    prompt = """
From the provided text extracted from a PDF document, your task is to identify and extract entities and their relationships as specified.

Entities to extract:
1. Organisation: The name of a company or organization.
2. Name: The name of a person.
3. PAN: The 10-character alphanumeric Permanent Account Number (format: 5 letters + 4 digits + 1 letter, e.g., AAUFM6247N).

Relation to extract:
- PAN_Of: This is the relationship between a PAN and the entity (Person or Organisation) it belongs to.

Instructions:
- Carefully analyze the text, focusing on tables that list noticees/names alongside their corresponding PANs.
- For every PAN and Name/Organisation pair you find, create a JSON object.
- The final output must be ONLY a JSON array containing all these objects, with no other text.
- Each object in the array should have three keys: 'pan', 'relation', and 'entity'.
- The value for 'relation' should always be the string "PAN_Of".
- Ensure the extracted PAN and entity names are accurate and exactly as they appear in the text.
- PANs are typically 10 characters: 5 letters, 4 digits, 1 letter (e.g., AAUFM6247N).
- Only include entries where both PAN and entity name are clearly identifiable.
- Do not include any markdown formatting, explanations, or code blocks - ONLY the JSON array.

Example Output Format:
[
  {
    "pan": "AAUFM6247N",
    "relation": "PAN_Of",
    "entity": "Mr. Agarwal"
  },
  {
    "pan": "AAACM9185B",
    "relation": "PAN_Of",
    "entity": "MAHESHWARI FINANCIAL SERVICES PVT. LTD."
  }
]

If no PAN-entity pairs are found, return: []
"""
    
    print(f"  Processing chunk {chunk_num}/{total_chunks}...")
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, chunk])
            
            # Check finish reason
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason != 1:  # 1 = STOP (normal completion)
                    print(f"    Warning: Chunk {chunk_num} finish_reason = {finish_reason}")
                    if finish_reason == 2:
                        print("    Chunk too large - may need smaller chunks")
                    elif finish_reason == 3:
                        print("    Content filtered for safety")
                    return []
            
            # Clean the response text
            cleaned_json_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if '```' in cleaned_json_text:
                import re
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned_json_text, re.DOTALL)
                if json_match:
                    cleaned_json_text = json_match.group(1)
                else:
                    cleaned_json_text = cleaned_json_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            parsed_data = json.loads(cleaned_json_text)
            
            if isinstance(parsed_data, list):
                print(f"    ✓ Found {len(parsed_data)} entities in chunk {chunk_num}")
                return parsed_data
            else:
                return [parsed_data] if parsed_data else []
                
        except json.JSONDecodeError as e:
            print(f"    JSON decode error in chunk {chunk_num} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"    ⚠️  Rate limit hit on chunk {chunk_num} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 4)  # 16, 32, 64 seconds
                    print(f"    Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"    ❌ Max retries reached for chunk {chunk_num}")
                    return []
            else:
                print(f"    Error processing chunk {chunk_num}: {e}")
                return []
    
    return []

def extract_entities_with_gemini(text: str, api_key: str) -> list[dict] | None:
    """Uses the Gemini API to extract entities and relations from text."""
    print("\nSending text to Gemini API for entity extraction...")
    try:
        # Get available models
        available_models = get_available_models(api_key)
        
        if not available_models:
            print("❌ No available models found. Please check your API key.")
            return None
        
        print(f"\nAvailable models: {len(available_models)}")
        print(f"Will try models in this order:")
        for i, model in enumerate(available_models[:5], 1):
            print(f"  {i}. {model}")
        
        # Try each model until one works
        for model_index, model_name in enumerate(available_models):
            try:
                print(f"\n{'='*60}")
                print(f"Attempting with model: {model_name}")
                print(f"{'='*60}")
                
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                )
                
                # Check if text is too large and needs chunking
                if len(text) > 120000:
                    print(f"\nDocument is large ({len(text)} chars). Processing in chunks...")
                    chunks = chunk_text(text, max_chars=120000)
                    print(f"Split into {len(chunks)} chunks.\n")
                    
                    all_entities = []
                    for i, chunk in enumerate(chunks, 1):
                        chunk_entities = extract_entities_from_chunk(model, chunk, i, len(chunks))
                        if chunk_entities:
                            all_entities.extend(chunk_entities)
                        
                        # Add delay between chunks to avoid rate limiting
                        if i < len(chunks):
                            print(f"    Waiting 5 seconds before next chunk...")
                            time.sleep(5)
                    
                    if all_entities:
                        print(f"\n✓ Total entities extracted: {len(all_entities)}")
                        return all_entities
                    elif model_index < len(available_models) - 1:
                        print(f"\n⚠️  No entities found with {model_name}, trying next model...")
                        continue
                    else:
                        return []
                else:
                    print("Processing document in single request...")
                    entities = extract_entities_from_chunk(model, text, 1, 1)
                    if entities:
                        return entities
                    elif model_index < len(available_models) - 1:
                        print(f"\n⚠️  No entities found with {model_name}, trying next model...")
                        continue
                    else:
                        return []
                        
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\n  Rate limit exceeded for {model_name}")
                    if model_index < len(available_models) - 1:
                        print(f"Trying next available model...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"\n All models exhausted. Please wait and try again later.")
                        print(f"Check your quota at: https://ai.dev/usage")
                        return None
                else:
                    print(f"\nError with {model_name}: {e}")
                    if model_index < len(available_models) - 1:
                        continue
                    else:
                        return None
        
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def write_to_csv(data: list[dict], csv_path: str):
    """Writes the extracted data to a CSV file."""
    if not data:
        print("\n  No data to write to CSV.")
        return
        
    print(f"\nWriting {len(data)} records to '{csv_path}'...")
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Entity (PAN)', 'Relation', 'Entity (Person)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in data:
                writer.writerow({
                    'Entity (PAN)': item.get('pan', ''),
                    'Relation': item.get('relation', ''),
                    'Entity (Person)': item.get('entity', '')
                })
        print(f"✓ Successfully wrote data to CSV file: {csv_path}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

def main():
    """Main function to run the entity extraction process."""
    # --- CONFIGURATION ---
    PDF_FILE_PATH = "toext.pdf" 
    OUTPUT_CSV_PATH = "result.csv"
    # ---------------------

    print("=" * 60)
    print("PDF Entity Extractor with Smart Retry")
    print("=" * 60)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(" Error: GOOGLE_API_KEY not found in .env file.")
        return
    
    # Step 1: Extract text from PDF
    pdf_text = extract_text_from_pdf(PDF_FILE_PATH)
    
    if not pdf_text:
        print("\n Could not extract text from PDF. Aborting.")
        return
    
    # Step 2: Extract entities using Gemini
    extracted_data = extract_entities_with_gemini(pdf_text, api_key)
    
    if extracted_data is None:
        print("\n Entity extraction failed.")
        print("\n Suggestions:")
        print("  1. Wait a few minutes and try again")
        print("  2. Check your quota: https://ai.dev/usage")
        print("  3. Consider upgrading to paid tier for higher limits")
        return
    
    if not extracted_data:
        print("\n  No PAN-entity pairs found in the document.")
    
    # Step 3: Write to CSV
    write_to_csv(extracted_data, OUTPUT_CSV_PATH)
    
    print("\n" + "=" * 60)
    print("✓ Process completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()