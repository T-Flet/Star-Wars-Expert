FROM python:3.11.7

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

COPY . .

WORKDIR /

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]


