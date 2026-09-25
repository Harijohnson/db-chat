-- chat-db schema
-- Paste this into the Supabase SQL editor (Dashboard -> SQL Editor -> New query)
-- and click "Run". Tables are created in order because of the foreign keys.

-- A program / course a student can belong to.
create table if not exists programs (
    id          bigint generated always as identity primary key,
    name        text not null,
    description text,
    created_at  timestamptz not null default now()
);

-- A student. NOTE: password is stored in plain text here for simplicity.
-- For anything real, hash it (e.g. bcrypt) instead of storing it directly.
create table if not exists students (
    id         bigint generated always as identity primary key,
    name       text not null,
    password   text not null,
    program_id bigint references programs (id) on delete set null,
    created_at timestamptz not null default now()
);

-- An event, optionally tied to a program. starts_at is a full timestamp.
create table if not exists events (
    id         bigint generated always as identity primary key,
    title      text,
    starts_at  timestamptz not null,
    program_id bigint references programs (id) on delete set null,
    created_at timestamptz not null default now()
);
