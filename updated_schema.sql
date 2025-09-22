-- Updated PostgreSQL Schema (Remove vector parts)
-- Keep all your existing tables, just remove resume_embeddings table

-- Drop the old vector table since we're moving to Qdrant
DROP TABLE IF EXISTS public.resume_embeddings;

-- Add vector tracking to main table
ALTER TABLE public.parsed_resumes
ADD COLUMN IF NOT EXISTS vector_id varchar(255) NULL,
ADD COLUMN IF NOT EXISTS embedding_status varchar(50) DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS embedding_model varchar(100) DEFAULT 'text-embedding-3-large',
ADD COLUMN IF NOT EXISTS embedding_generated_at timestamptz NULL;

-- Index for vector tracking
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_vector_id ON public.parsed_resumes(vector_id);
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_embedding_status ON public.parsed_resumes(embedding_status);

-- Enhanced search indexes for PostgreSQL portion
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_name_gin ON public.parsed_resumes USING gin(to_tsvector('english', name));
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_skills_gin ON public.parsed_resumes USING gin(skills);
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_location_gin ON public.parsed_resumes USING gin(to_tsvector('english', location));
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_summary_gin ON public.parsed_resumes USING gin(to_tsvector('english', summary));
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_current_position_gin ON public.parsed_resumes USING gin(to_tsvector('english', current_position));

-- Full text search index across all text fields
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_full_text ON public.parsed_resumes
USING gin(to_tsvector('english',
    COALESCE(name, '') || ' ' ||
    COALESCE(summary, '') || ' ' ||
    COALESCE(current_position, '') || ' ' ||
    COALESCE(location, '') || ' ' ||
    COALESCE(original_text, '')
));

-- Add trigger to update embedding_status
CREATE OR REPLACE FUNCTION update_embedding_status()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.vector_id IS NOT NULL AND OLD.vector_id IS NULL THEN
        NEW.embedding_status = 'completed';
        NEW.embedding_generated_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_embedding_status
    BEFORE UPDATE ON public.parsed_resumes
    FOR EACH ROW
    EXECUTE FUNCTION update_embedding_status();