this is  astable version of demo hosted in 7777 port
# AI-Powered Chatbot with FAQ Management

A Flask-based chatbot application that uses OpenAI's GPT model to answer questions based on uploaded documents and FAQs. The system includes an admin interface for managing FAQs, documents, and user queries.

## Prerequisites

- Python 3.8 or higher
- Git (optional, for version control)
- OpenAI API key

## Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd thesis-rag
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root with the following variables:
   ```
   # Required Settings
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=a1d532a27fe5cd6b773e24b036b8590c82e6a3ebf0e8ac93b8e8de7528e24674
   
   # Optional Settings
   APP_NAME=RagBook
   MAX_CONTENT_LENGTH=16777216  # 16MB max file size
   ```
   
   Note: 
   - Generate a secure secret key using Python:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```
   - In production, set `FLASK_ENV=production`
   - OpenAI API key should be configured in the admin settings after installation

5. **Initialize the Database**
   ```bash
   python init_db.py
   ```
   This will:
   - Create the database tables
   - Create an admin user (email: admin@gmail.com, password: 123456)
   - Set up initial application settings

## Running the Application

1. **Development Mode**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:8080`

2. **Production Mode**
   ```bash
   export FLASK_ENV=production
   python app.py
   ```

## Project Structure

```
project/
├── app.py              # Application factory
├── config.py           # Configuration classes
├── init_db.py          # Database initialization
├── models.py           # Database models
├── routes.py           # Route handlers
├── embed_and_search.py # Embedding and search functionality
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
└── templates/         # Template files
    ├── admin/        # Admin interface templates
    └── public/       # Public interface templates
```

## First-Time Setup

1. **Access the Admin Interface**
   - Go to `http://localhost:5000/admin/login`
   - Login with:
     - Email: admin@gmail.com
     - Password: 123456

2. **Configure Settings**
   - Go to Settings in the admin panel
   - Update the OpenAI API key
   - Customize application settings (logo, about, contact info)

3. **Add FAQs**
   - Go to FAQs in the admin panel
   - Add your frequently asked questions and answers
   - Mark answers as AI-generated if appropriate

4. **Upload Documents**
   - Go to Files in the admin panel
   - Upload documents (PDF, DOCX, TXT)
   - Add web pages or YouTube videos by URL
   - Paste raw text directly

## Features

- **Chat Interface**
  - Real-time question answering
  - Source citation for answers
  - Fallback to new questions when no answer is found

- **Admin Dashboard**
  - FAQ management
  - Document management
  - User management
  - Query history
  - New questions review
  - System settings

- **Security**
  - User authentication
  - Role-based access control
  - Secure password handling
  - Environment variable protection

## Troubleshooting

1. **Database Issues**
   - If you encounter database errors, try:
     ```bash
     rm chatbot.db
     python init_db.py
     ```

2. **OpenAI API Issues**
   - Verify your API key in the admin settings
   - Check your internet connection
   - Ensure you have sufficient API credits

3. **File Upload Issues**
   - Check file size limits
   - Verify file format support
   - Ensure proper permissions

## Security Notes

- Change the default admin password after first login
- Keep your `.env` file secure and never commit it to version control
- Regularly update dependencies for security patches
- Use strong passwords for all user accounts

## Updating the Application

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Update Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Run Database Migrations** (if any)
   ```bash
   flask db upgrade
   ```

## Support

For issues and feature requests, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

## License

[Your License Here] 