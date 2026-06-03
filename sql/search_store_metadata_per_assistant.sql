SELECT prefix, 
	value -> 'document' -> 'kwargs' -> 'metadata' as full_metadata, 
    (value->'document'->'kwargs'->'metadata'->>'filename') as filename, 
    (value->'document'->'kwargs'->'metadata'->>'assistant_id') as assistant_id 
 FROM store 
 WHERE prefix LIKE '%bf40fb38-59b9-4045-9e2f-09607a9578e6%' 
 LIMIT 100