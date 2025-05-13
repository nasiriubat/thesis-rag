# AI-Powered Chatbot with Admin Dashboard

A responsive web application featuring a public chatbot interface and an admin dashboard for managing content and settings.

## Features

- **Public Chatbot Interface**
  - Responsive design with sidebar and main chat area
  - FAQ cards and scrollable FAQ list
  - Real-time chat with AI-powered responses
  - Dynamic content from admin settings

- **Admin Dashboard**
  - Secure login system
  - Full CRUD operations for:
    - User management
    - Role management
    - FAQ management
    - File uploads (PDF, DOCX, text, URLs)
    - Query history
    - System settings

- **AI Integration**
  - Text embedding using SentenceTransformer
  - Semantic search across uploaded documents
  - OpenAI integration for generating responses
  - Relevance scoring for better answers

## Prerequisites

- Python 3.8+
- OpenAI API key
- SQLite (included with Python)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chatbot-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   SECRET_KEY=your-secret-key-here
   OPENAI_API_KEY=your-openai-api-key
   ```

5. Initialize the database:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Access the application:
   - Public interface: http://localhost:5000
   - Admin dashboard: http://localhost:5000/admin
   - Default admin credentials:
     - Email: admin@gmail.com
     - Password: 123456

## Project Structure

```
chatbot-app/
├── app.py              # Main application file
├── routes.py           # Route definitions
├── embed_and_search.py # Embedding and search functionality
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── dataembedding/     # Embedding storage
├── templates/         # HTML templates
│   ├── base.html
│   ├── public/
│   └── admin/
└── static/           # Static files (CSS, JS, images)
```

## Security Notes

- Change the default admin password after first login
- Keep your OpenAI API key secure
- Use environment variables for sensitive data
- Regular security updates recommended

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 