-- PRISM assessment persistence (Phase 1)
-- Run this in the Supabase SQL editor after creating a project.

create table if not exists assessments (
  id uuid primary key default gen_random_uuid(),
  name text not null default 'Untitled',
  share_code text unique not null,
  owner_label text default 'Lead',
  state_json jsonb not null default '{}'::jsonb,
  status text not null default 'draft'
    check (status in ('draft', 'in_progress', 'complete')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_assessments_share_code on assessments (share_code);
create index if not exists idx_assessments_updated_at on assessments (updated_at desc);

create table if not exists category_assignments (
  id uuid primary key default gen_random_uuid(),
  assessment_id uuid not null references assessments (id) on delete cascade,
  category_key text not null
    check (category_key in ('env', 'eco', 'soc', 'qua', 'pro')),
  assignee_name text not null default '',
  status text not null default 'pending'
    check (status in ('pending', 'in_progress', 'complete')),
  updated_at timestamptz not null default now(),
  unique (assessment_id, category_key)
);

create index if not exists idx_category_assignments_assessment
  on category_assignments (assessment_id);

-- Phase 1: permissive policies for anon key (app validates share codes)
alter table assessments enable row level security;
alter table category_assignments enable row level security;

create policy "anon_all_assessments" on assessments
  for all using (true) with check (true);

create policy "anon_all_category_assignments" on category_assignments
  for all using (true) with check (true);
