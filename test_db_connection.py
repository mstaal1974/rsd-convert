#!/usr/bin/env python3
"""
Test database connection to Neon PostgreSQL
"""
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env file")
    exit(1)

print(f"Testing connection to database...")
print(f"Connection string: {DATABASE_URL[:80]}...\n")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Test query
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    
    print("✓ Connection successful!")
    print(f"Server version: {db_version[0]}")
    
    # Get database info
    cursor.execute("SELECT current_database(), current_user;")
    db_info = cursor.fetchone()
    print(f"Database: {db_info[0]}")
    print(f"User: {db_info[1]}")
    
    cursor.close()
    conn.close()
    
except psycopg2.Error as e:
    print(f"✗ Connection failed!")
    print(f"Error: {e}")
    exit(1)
