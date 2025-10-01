-- Create basic tables

CREATE SCHEMA IF NOT EXISTS public;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS public.users (
    id varchar(255) NOT NULL,
    email varchar(255) NOT NULL,
    "name" varchar(255) NULL,
    image varchar(500) NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    CONSTRAINT users_email_key UNIQUE (email),
    CONSTRAINT users_pkey PRIMARY KEY (id)
);

-- Parsed resumes table (as per ddl.sql)
CREATE TABLE IF NOT EXISTS public.parsed_resumes (
    id uuid DEFAULT uuid_generate_v4() NOT NULL,
    "name" text NULL,
    email text NULL,
    phone varchar(50) NULL,
    "location" text NULL,
    linkedin_url text NULL,
    skills jsonb DEFAULT '[]'::jsonb NULL,
    education jsonb DEFAULT '[]'::jsonb NULL,
    experience jsonb DEFAULT '[]'::jsonb NULL,
    projects jsonb DEFAULT '[]'::jsonb NULL,
    summary text NULL,
    total_experience varchar(100) NULL,
    recommended_roles jsonb DEFAULT '[]'::jsonb NULL,
    role_classification jsonb DEFAULT '{}'::jsonb NULL,
    best_role text NULL,
    processed_at timestamptz DEFAULT now() NULL,
    "source" varchar(50) DEFAULT 'email'::character varying NULL,
    created_at timestamptz DEFAULT now() NULL,
    content_hash varchar(64) NULL,
    current_position text NULL,
    user_id varchar(255) NULL,
    original_text text NULL,
    is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT parsed_resumes_pkey PRIMARY KEY (id),
    CONSTRAINT parsed_resumes_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL
);

-- Qdrant resumes table (as used in code)
CREATE TABLE IF NOT EXISTS public.qdrant_resumes (
    id uuid PRIMARY KEY,
    vector_id varchar(255),
    owner_user_id varchar(255),
    name text,
    email text,
    phone varchar(50),
    location text,
    linkedin_url text,
    current_position text,
    summary text,
    total_experience text,
    role_category text,
    seniority text,
    best_role text,
    skills jsonb DEFAULT '[]'::jsonb,
    work_history jsonb DEFAULT '[]'::jsonb,
    projects jsonb DEFAULT '[]'::jsonb,
    education jsonb DEFAULT '[]'::jsonb,
    role_classification jsonb DEFAULT '{}'::jsonb,
    recommended_roles jsonb DEFAULT '[]'::jsonb,
    original_filename text,
    upload_timestamp timestamptz,
    embedding_model varchar(100) DEFAULT 'text-embedding-3-large',
    embedding_generated_at timestamptz,
    created_at timestamptz DEFAULT now(),
    is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE
);

-- Interview questions
CREATE TABLE IF NOT EXISTS public.interview_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    question_text TEXT NOT NULL,
    expected_answer TEXT,
    welcome_message TEXT,
    category VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

-- Interviews
CREATE TABLE IF NOT EXISTS public.interviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    welcome_message TEXT,
    question_ids UUID[] NOT NULL,
    candidate_ids UUID[],
    status VARCHAR(50) DEFAULT 'draft' NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_interview_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

-- Call records
CREATE TABLE IF NOT EXISTS public.call_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resume_id UUID NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    call_status VARCHAR(50) NOT NULL,
    initiated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT fk_resume FOREIGN KEY (resume_id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

-- User resume limits
CREATE TABLE IF NOT EXISTS public.user_resume_limits (
    id uuid DEFAULT uuid_generate_v4() NOT NULL,
    user_id varchar(255) NOT NULL,
    total_resumes_uploaded int4 DEFAULT 0 NULL,
    resume_limit int4 DEFAULT 10 NULL,
    tokens_used int8 DEFAULT 0 NULL,
    token_limit int8 DEFAULT 1000000 NULL,
    created_at timestamptz DEFAULT now() NULL,
    updated_at timestamptz DEFAULT now() NULL,
    last_resume_uploaded_at timestamptz NULL,
    CONSTRAINT check_non_negative_counts CHECK (((total_resumes_uploaded >= 0) AND (tokens_used >= 0) AND (resume_limit > 0) AND (token_limit > 0))),
    CONSTRAINT user_resume_limits_pkey PRIMARY KEY (id),
    CONSTRAINT user_resume_limits_user_id_key UNIQUE (user_id),
    CONSTRAINT user_resume_limits_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

-- Interview transcripts
CREATE TABLE IF NOT EXISTS public.interview_transcripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    question_id UUID NOT NULL,
    candidate_response TEXT,
    ai_evaluation TEXT,
    audio_file_path TEXT,
    audio_duration FLOAT,
    response_time_seconds FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_transcript_session FOREIGN KEY (session_id) REFERENCES public.interview_sessions(id) ON DELETE CASCADE,
    CONSTRAINT fk_transcript_question FOREIGN KEY (question_id) REFERENCES public.interview_questions(id) ON DELETE CASCADE
);

-- Interviewer decisions
CREATE TABLE IF NOT EXISTS public.interviewer_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    resume_id UUID NOT NULL,
    decision VARCHAR(50) NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_decision_session FOREIGN KEY (session_id) REFERENCES public.interview_sessions(id) ON DELETE CASCADE,
    CONSTRAINT fk_decision_resume FOREIGN KEY (resume_id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE
);

-- Interview responses
CREATE TABLE IF NOT EXISTS public.interview_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    question_id UUID NOT NULL,
    answer_text TEXT,
    audio_file_path TEXT,
    audio_duration FLOAT,
    response_time_seconds FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_user_responses FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE,
    CONSTRAINT fk_question_responses FOREIGN KEY (question_id) REFERENCES public.interview_questions(id) ON DELETE CASCADE
);

-- User search prompts
CREATE TABLE IF NOT EXISTS public.user_search_prompts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    prompt_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_search_prompt_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_user_id ON public.parsed_resumes(user_id);
CREATE INDEX IF NOT EXISTS idx_parsed_resumes_is_shortlisted ON public.parsed_resumes(is_shortlisted);
CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_owner ON public.qdrant_resumes(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_is_shortlisted ON public.qdrant_resumes(is_shortlisted);
CREATE INDEX IF NOT EXISTS idx_interview_questions_user_id ON public.interview_questions(user_id);
CREATE INDEX IF NOT EXISTS idx_interviews_user_id ON public.interviews(user_id);
CREATE INDEX IF NOT EXISTS idx_call_records_resume_id ON public.call_records(resume_id);
CREATE INDEX IF NOT EXISTS idx_call_records_user_id ON public.call_records(user_id);
CREATE INDEX IF NOT EXISTS idx_user_resume_limits_user_id ON public.user_resume_limits USING btree (user_id);
CREATE INDEX IF NOT EXISTS idx_interview_transcripts_session_id ON public.interview_transcripts(session_id);
CREATE INDEX IF NOT EXISTS idx_interview_transcripts_question_id ON public.interview_transcripts(question_id);
CREATE INDEX IF NOT EXISTS idx_interviewer_decisions_session_id ON public.interviewer_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_interviewer_decisions_resume_id ON public.interviewer_decisions(resume_id);
CREATE INDEX IF NOT EXISTS idx_interview_responses_user_id ON public.interview_responses(user_id);
CREATE INDEX IF NOT EXISTS idx_interview_responses_question_id ON public.interview_responses(question_id);
CREATE INDEX IF NOT EXISTS idx_interview_responses_created_at ON public.interview_responses(created_at);
CREATE INDEX IF NOT EXISTS idx_user_search_prompts_user_id ON public.user_search_prompts(user_id);