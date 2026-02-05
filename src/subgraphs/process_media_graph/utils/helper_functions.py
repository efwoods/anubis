from typing import Any

import base64

import logging
logger = logging.getLogger(__name__)


def identify_file_type_and_convert_to_base64(media_files: list[Any]): 
    media_list = []
    for file_data in media_files:
        try:
           # Extract file info
           filename = file_data.get('filename', 'unknown')
           content_type = file_data.get('content_type', '')
           file_bytes = file_data.get('content')  # Raw bytes
           
           logger.info(f"Processing file: {filename} ({content_type})")
           
           # Determine media type and convert to standardized format
           if content_type.startswith('image/'):
               # Convert image to base64
               base64_data = base64.b64encode(file_bytes).decode('utf-8')
               media_list.append({
                   "type": "image",
                   "data": base64_data,
                   "metadata": {
                       "filename": filename,
                       "content_type": content_type,
                       "size": len(file_bytes)
                   }
               })
           
           elif content_type.startswith('audio/'):
               # Handle audio files
               base64_data = base64.b64encode(file_bytes).decode('utf-8')
               media_list.append({
                   "type": "audio",
                   "data": base64_data,
                   "metadata": {
                       "filename": filename,
                       "content_type": content_type,
                       "size": len(file_bytes)
                   }
               })
           
           elif content_type.startswith('video/'):
               # Handle video files
               base64_data = base64.b64encode(file_bytes).decode('utf-8')
               media_list.append({
                   "type": "video",
                   "data": base64_data,
                   "metadata": {
                       "filename": filename,
                       "content_type": content_type,
                       "size": len(file_bytes)
                   }
               })
           
           elif content_type in ['text/plain', 'application/json', 'text/markdown']:
               # Handle text files
               text_content = file_bytes.decode('utf-8')
               media_list.append({
                   "type": "text",
                   "content": text_content,
                   "metadata": {
                       "filename": filename,
                       "content_type": content_type,
                       "size": len(file_bytes)
                   }
               })
           
           elif content_type == 'application/pdf':
               # Handle PDFs
               base64_data = base64.b64encode(file_bytes).decode('utf-8')
               media_list.append({
                   "type": "pdf",
                   "data": base64_data,
                   "metadata": {
                       "filename": filename,
                       "content_type": content_type,
                       "size": len(file_bytes)
                   }
               })
           
           else:
               logger.warning(f"Unsupported content type: {content_type}")
               continue
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Converted {len(media_list)} files to media format")
    