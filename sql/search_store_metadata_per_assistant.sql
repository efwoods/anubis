SELECT prefix, 
	value -> 'document' -> 'kwargs' -> 'metadata' as full_metadata, 
    (value->'document'->'kwargs'->'metadata'->>'filename') as filename, 
    (value->'document'->'kwargs'->'metadata'->>'assistant_id') as assistant_id 
 FROM store 
 WHERE prefix LIKE '%e1699cb2-cbc2-4c92-966c-2c830b0df910%' 
 LIMIT 100