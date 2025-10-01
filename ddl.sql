-- public.odoo_job_submissions definition

-- Drop table

-- DROP TABLE public.odoo_job_submissions;

CREATE TABLE public.odoo_job_submissions (
	submitted_at timestamptz NULL,
	company_id int4 NULL,
	odoo_applicant_id int4 NULL,
	parsed_resume_id uuid NULL,
	stage_id int4 NULL,
	user_id int4 NULL,
	id uuid NULL,
	job_title varchar(255) NULL,
	applicant_notes text NULL,
	availability varchar(100) NULL,
	submission_status varchar(50) NULL,
	error_message text NULL
);


-- public.users definition

-- Drop table

-- DROP TABLE public.users;

CREATE TABLE public.users (
	id varchar(255) NOT NULL,
	email varchar(255) NOT NULL,
	"name" varchar(255) NULL,
	image varchar(500) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT users_email_key UNIQUE (email),
	CONSTRAINT users_pkey PRIMARY KEY (id)
);


-- public.parsed_resumes definition

-- Drop table

-- DROP TABLE public.parsed_resumes;

CREATE TABLE public.parsed_resumes (
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
	CONSTRAINT parsed_resumes_pkey PRIMARY KEY (id),
	CONSTRAINT parsed_resumes_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL
);
CREATE INDEX idx_parsed_resumes_content_hash ON public.parsed_resumes USING btree (content_hash);
CREATE INDEX idx_parsed_resumes_created_at ON public.parsed_resumes USING btree (created_at);
CREATE INDEX idx_parsed_resumes_email ON public.parsed_resumes USING btree (email);
CREATE INDEX idx_parsed_resumes_user_id ON public.parsed_resumes USING btree (user_id);


-- public.resume_embeddings definition

-- Drop table

-- DROP TABLE public.resume_embeddings;

CREATE TABLE public.resume_embeddings (
	id uuid NOT NULL,
	resume_embedding public.vector NULL,
	CONSTRAINT resume_embeddings_pkey PRIMARY KEY (id),
	CONSTRAINT resume_embeddings_id_fkey FOREIGN KEY (id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE
);


-- public.resume_files definition

-- Drop table

-- DROP TABLE public.resume_files;

CREATE TABLE public.resume_files (
	id uuid NOT NULL,
	file_base64 text NULL,
	original_filename text NULL,
	file_type varchar(10) NULL,
	CONSTRAINT resume_files_pkey PRIMARY KEY (id),
	CONSTRAINT resume_files_id_fkey FOREIGN KEY (id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE
);


-- public.smart_screening_prompt definition

-- Drop table

-- DROP TABLE public.smart_screening_prompt;

CREATE TABLE public.smart_screening_prompt (
	id uuid DEFAULT uuid_generate_v4() NOT NULL,
	user_id varchar(255) NOT NULL,
	user_prompt text NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT smart_screening_prompt_pkey PRIMARY KEY (id),
	CONSTRAINT smart_screening_prompt_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);
CREATE INDEX idx_smart_screening_prompt_created_at ON public.smart_screening_prompt USING btree (created_at);
CREATE INDEX idx_smart_screening_prompt_user_id ON public.smart_screening_prompt USING btree (user_id);


-- public.user_microsoft_tokens definition

-- Drop table

-- DROP TABLE public.user_microsoft_tokens;

CREATE TABLE public.user_microsoft_tokens (
	id bigserial NOT NULL,
	user_id varchar(255) NOT NULL,
	access_token text NOT NULL,
	refresh_token text NOT NULL,
	expires_at timestamp NOT NULL,
	token_type varchar(50) DEFAULT 'Bearer'::character varying NULL,
	"scope" text NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT user_microsoft_tokens_pkey PRIMARY KEY (id),
	CONSTRAINT user_microsoft_tokens_user_id_key UNIQUE (user_id),
	CONSTRAINT user_microsoft_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);


-- public.user_resume_limits definition

-- Drop table

-- DROP TABLE public.user_resume_limits;

CREATE TABLE public.user_resume_limits (
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
CREATE INDEX idx_user_resume_limits_user_id ON public.user_resume_limits USING btree (user_id);

-- Table Triggers

create trigger update_user_resume_limits_updated_at before
update
    on
    public.user_resume_limits for each row execute function update_user_resume_limits_updated_at();


-- Drop tables in reverse order of dependency to avoid foreign key constraints issues
DROP TABLE IF EXISTS public.call_records;
DROP TABLE IF EXISTS public.interview_questions;
DROP TABLE IF EXISTS public.smart_screening_prompt;
DROP TABLE IF EXISTS public.resume_files;
DROP TABLE IF EXISTS public.resume_embeddings;
DROP TABLE IF EXISTS public.parsed_resumes;
DROP TABLE IF EXISTS public.user_resume_limits;
DROP TABLE IF EXISTS public.user_microsoft_tokens;
DROP TABLE IF EXISTS public.odoo_job_submissions;
DROP TABLE IF EXISTS public.users;

-- Enable UUID generation extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- public.users definition
CREATE TABLE public.users (
	id varchar(255) NOT NULL,
	email varchar(255) NOT NULL,
	"name" varchar(255) NULL,
	image varchar(500) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT users_email_key UNIQUE (email),
	CONSTRAINT users_pkey PRIMARY KEY (id)
);

-- public.parsed_resumes definition
CREATE TABLE public.parsed_resumes (
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
    is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE, -- New column for shortlisting
	CONSTRAINT parsed_resumes_pkey PRIMARY KEY (id),
	CONSTRAINT parsed_resumes_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL
);

CREATE INDEX idx_parsed_resumes_content_hash ON public.parsed_resumes USING btree (content_hash);
CREATE INDEX idx_parsed_resumes_created_at ON public.parsed_resumes USING btree (created_at);
CREATE INDEX idx_parsed_resumes_email ON public.parsed_resumes USING btree (email);
CREATE INDEX idx_parsed_resumes_user_id ON public.parsed_resumes USING btree (user_id);
CREATE INDEX idx_parsed_resumes_is_shortlisted ON public.parsed_resumes(is_shortlisted); -- Index for shortlisting

-- New table for custom interview questions
CREATE TABLE public.interview_questions (
    id uuid DEFAULT uuid_generate_v4() NOT NULL,
    user_id varchar(255) NOT NULL,
    question_text text NOT NULL,
    expected_answer text,
    welcome_message text,
    category varchar(100),
    created_at timestamptz DEFAULT now() NOT NULL,
    updated_at timestamptz DEFAULT now() NOT NULL,
    CONSTRAINT interview_questions_pkey PRIMARY KEY (id),
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

CREATE INDEX idx_interview_questions_user_id ON public.interview_questions(user_id);

-- New table for interview sessions
CREATE TABLE public.interviews (
    id uuid DEFAULT uuid_generate_v4() NOT NULL,
    user_id varchar(255) NOT NULL,
    title text NOT NULL,
    description text,
    welcome_message text,
    question_ids uuid[] NOT NULL,
    candidate_ids uuid[],
    status varchar(50) DEFAULT 'draft' NOT NULL, -- 'draft', 'active', 'completed', 'cancelled'
    created_at timestamptz DEFAULT now() NOT NULL,
    updated_at timestamptz DEFAULT now() NOT NULL,
    CONSTRAINT interviews_pkey PRIMARY KEY (id),
    CONSTRAINT fk_interview_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

CREATE INDEX idx_interviews_user_id ON public.interviews(user_id);
CREATE INDEX idx_interviews_status ON public.interviews(status);

-- New table to log interview call records
CREATE TABLE public.call_records (
    id uuid DEFAULT uuid_generate_v4() NOT NULL,
    resume_id uuid NOT NULL,
    user_id varchar(255) NOT NULL,
    call_status varchar(50) NOT NULL, -- e.g., 'initiated', 'completed', 'failed'
    initiated_at timestamptz DEFAULT now() NOT NULL,
    notes text,
    CONSTRAINT call_records_pkey PRIMARY KEY (id),
    CONSTRAINT fk_resume FOREIGN KEY (resume_id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

CREATE INDEX idx_call_records_resume_id ON public.call_records(resume_id);
CREATE INDEX idx_call_records_user_id ON public.call_records(user_id);


-- Other tables from your original DDL
CREATE TABLE public.odoo_job_submissions (
	submitted_at timestamptz NULL,
	company_id int4 NULL,
	odoo_applicant_id int4 NULL,
	parsed_resume_id uuid NULL,
	stage_id int4 NULL,
	user_id int4 NULL,
	id uuid NULL,
	job_title varchar(255) NULL,
	applicant_notes text NULL,
	availability varchar(100) NULL,
	submission_status varchar(50) NULL,
	error_message text NULL
);


CREATE TABLE public.resume_files (
	id uuid NOT NULL,
	file_base64 text NULL,
	original_filename text NULL,
	file_type varchar(10) NULL,
	CONSTRAINT resume_files_pkey PRIMARY KEY (id),
	CONSTRAINT resume_files_id_fkey FOREIGN KEY (id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE
);


CREATE TABLE public.smart_screening_prompt (
	id uuid DEFAULT uuid_generate_v4() NOT NULL,
	user_id varchar(255) NOT NULL,
	user_prompt text NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT smart_screening_prompt_pkey PRIMARY KEY (id),
	CONSTRAINT smart_screening_prompt_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);
CREATE INDEX idx_smart_screening_prompt_created_at ON public.smart_screening_prompt USING btree (created_at);
CREATE INDEX idx_smart_screening_prompt_user_id ON public.smart_screening_prompt USING btree (user_id);


CREATE TABLE public.user_microsoft_tokens (
	id bigserial NOT NULL,
	user_id varchar(255) NOT NULL,
	access_token text NOT NULL,
	refresh_token text NOT NULL,
	expires_at timestamp NOT NULL,
	token_type varchar(50) DEFAULT 'Bearer'::character varying NULL,
	"scope" text NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT user_microsoft_tokens_pkey PRIMARY KEY (id),
	CONSTRAINT user_microsoft_tokens_user_id_key UNIQUE (user_id),
	CONSTRAINT user_microsoft_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);

CREATE TABLE public.user_resume_limits (
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
CREATE INDEX idx_user_resume_limits_user_id ON public.user_resume_limits USING btree (user_id);
