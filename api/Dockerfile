# 1. Use the Playwright Python image whose tag MATCHES the playwright version
#    pinned in requirements.txt (playwright==1.61.0). A mismatched tag makes
#    Crawl4AI download a newer Chromium at runtime whose OS libraries
#    (libglib-2.0.so.0, etc.) are missing from the image.
FROM mcr.microsoft.com/playwright/python:v1.61.0-noble

# 2. Set the working directory
WORKDIR /api

# 3. Copy your requirements and install Python dependencies
# This installs the 'playwright' Python package, which allows your code to
# communicate with the browsers already present in the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install the Chromium browser AND its OS-level shared libraries for the
#    installed Playwright version. `--with-deps` apt-installs libglib-2.0.so.0
#    and the rest system-wide, so the browser runs regardless of cache path.
RUN playwright install --with-deps chromium

# 5. Copy the rest of your project code
COPY . .

# 6. Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
