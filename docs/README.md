# CommServe - Intelligent E-Commerce Data Analysis Platform

## What is CommServe?

**CommServe** is an intelligent conversational AI platform that transforms how businesses interact with their e-commerce data. Instead of writing complex SQL queries or struggling with data analysis tools, you can simply **ask questions in plain English** and get instant insights, visualizations, and actionable recommendations.

### The Problem It Solves

Traditional data analysis requires:

- Writing complex SQL queries
- Learning specialized BI tools
- Waiting for IT teams to generate reports
- Manual data cleaning and preparation

**CommServe changes this completely:**

- **Natural language queries**: "Show me sales trends for the last quarter"
- **Instant responses**: Get answers in seconds, not hours
- **Self-service analytics**: Business users can explore data independently
- **Automated insights**: AI-generated recommendations and patterns

---

## Complete Setup Guide

### Prerequisites

- Python 3.8 or higher
- Git
- At least one API key (OpenAI, Gemini, or OpenRouter)

### Step 1: Clone and Environment Setup

```bash
git clone https://github.com/Achlys2004/CommServe.git
cd CommServe
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # Linux/Mac
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the root directory:

```bash
# Copy and edit the following:
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

DATABASE_URL=sqlite:///./data/olist.db
CHROMA_DB_DIR=./embeddings/chroma
USE_INTELLIGENT_PLANNER=true

FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000
```

**Note**: You need at least one API key. Get them from:

- [OpenAI](https://platform.openai.com/api-keys)
- [Google AI Studio](https://makersuite.google.com/app/apikey) (for Gemini)
- [OpenRouter](https://openrouter.ai/keys)

### Step 4: Download and Prepare Data

CommServe uses the Brazilian E-Commerce dataset from Kaggle. You need to download and prepare the data:

#### Download the Dataset

1. Visit the [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Download the dataset (ZIP file)
3. Extract the 9 CSV files to the `data/` folder in your project

**Required CSV files:**

- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

#### Clean and Load Data

Once the CSV files are in place, run the data preparation scripts:

```bash
# Clean and load data into SQLite database
python scripts/load_cleaned_data.py

# Add database indexes for performance
python scripts/add_database_indexes.py

# Validate data integrity
python scripts/check_schema.py
```

**What this creates:**

- `data/olist.db` - SQLite database with cleaned data
- Database indexes for fast queries
- Validation reports

### Step 5: Build Embeddings for RAG

Create vector embeddings for semantic search:

```bash
# Build ChromaDB embeddings from product data
python scripts/build_embeddings.py
```

**What this creates:**

- `embeddings/chroma/` - Vector database for RAG functionality
- Embeddings for product descriptions and reviews

### Step 6: Launch the Backend

Start the FastAPI backend server:

```bash
# In one terminal (keep running)
uvicorn backend.app:app --reload
```

**What this provides:**

- API endpoints at http://localhost:8000
- Interactive API docs at http://localhost:8000/docs
- Query processing and AI integration

### Step 7: Launch the Frontend

Start the Streamlit chat interface:

```bash
# In another terminal (keep running)
streamlit run frontend/chat_ui.py
```

**What this provides:**

- Web interface at http://localhost:8501
- Conversational AI chat for data analysis

### Step 8: Start Exploring!

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Try your first query**: _"What are our best-selling product categories?"_

---

## Implementation Details

CommServe started as a simple data analysis tool and evolved into a sophisticated AI-powered business intelligence platform. Here's how it grew:

#### Phase 0: Research & Planning (day 1)

- Researched suitable architectures, methods, and techniques for AI-powered data analysis
- Created comprehensive research documents in `research/` directory
- Explored multiple architecture variations for GenAI agents
- Analyzed agentic architectures with planner-executor patterns
- Evaluated RAG (Retrieval-Augmented Generation) implementations
- Studied memory and storage strategies
- Developed self-evaluation mechanisms
- Selected planner-executor-RAG architecture for CommServe
- Defined component interactions and data flows
- Established evaluation criteria for different approaches

#### Phase 1: Data Foundation (day 2)

- Established reliable data pipeline
- Handled messy e-commerce data with duplicates, missing values, and inconsistent formats
- Built robust data cleaning pipeline in `scripts/load_cleaned_data.py`
- Implemented SQLite database with optimized schema
- Added data validation and integrity checks

#### Phase 2: AI Integration (day 3)

- Added intelligent query processing based on research findings
- Made AI understand business context and generate accurate SQL
- Implemented multi-tier LLM fallback (OpenAI → Gemini → OpenRouter)
- Built intelligent query planner that understands business metrics
- Added RAG system with ChromaDB for context-aware responses

#### Phase 3: User Experience (day 4)

- Created intuitive interface for non-technical users
- Translated complex data analysis into conversational interactions
- Developed Streamlit frontend with modern chat interface
- Built FastAPI backend for high-performance API
- Implemented real-time query execution and result visualization

#### Phase 4: Production Readiness (day 5)

- Made system reliable and maintainable
- Implemented error handling, testing, and documentation
- Created comprehensive test suite (34 tests covering all features)
- Added graceful dependency handling and fallback mechanisms
- Developed detailed documentation and setup guides

#### Phase 5: Wrapping up the project and Document Preparation (day 6)

- Finalized comprehensive documentation including setup guides, architecture details, and user manuals
- Conducted thorough testing and validation of all features with edge cases

### Architecture Decisions

#### Why SQLite?

- **Simplicity**: No external database setup required
- **Performance**: Excellent for read-heavy analytical workloads
- **Portability**: Single file database, easy to backup and share

#### Why Multi-Tier LLM Fallback?

- **Reliability**: If one provider fails, others automatically take over
- **Cost Optimization**: Use cheaper providers when available
- **Future-Proofing**: Easy to add new LLM providers

#### Why ChromaDB for Embeddings?

- **Local**: No cloud dependency for vector storage
- **Fast**: Optimized for similarity search
- **Simple**: Easy integration with existing Python stack

### Future Roadmap

- **Advanced Analytics**: Time-series analysis and predictive modeling
- **Multi-Language Support**: Support for non-English datasets
- **Real-time Dashboards**: Live data visualization and monitoring
- **API Rate Limiting**: Better handling of API quotas and costs
- **Plugin Architecture**: Extensible system for custom analysis modules

---

## Features & Capabilities

### Natural Language Processing

- **Multi-language support**: Ask questions in English or Portuguese
- **Context awareness**: Follow-up questions like "What about last month?" work perfectly
- **Intelligent query understanding**: No need for perfect grammar or technical terms

### Data Analysis Modes

#### SQL Mode

- Generates and executes SQL queries automatically
- Handles complex joins, aggregations, and filtering
- Self-validates queries before execution
- Supports all SQLite operations

#### Code Generation Mode

- Creates Python scripts for advanced analysis
- Generates matplotlib/seaborn visualizations
- Performs statistical analysis
- Auto-executes code and displays results

#### RAG (Retrieval-Augmented Generation)

- Searches through documentation and knowledge base
- Provides contextual answers about your business
- Combines data insights with business knowledge

#### Hybrid Mode

- Combines multiple analysis approaches
- Provides comprehensive answers with data + insights
- Best for complex business questions

### Safety Features

#### Multi-Tier AI Fallback

**Automatic failover system** with intelligent routing:

- **Tier 1**: OpenAI GPT-4 (Primary) - Multiple API keys with load balancing
- **Tier 2**: Google Gemini (Backup) - Automatic activation on Tier 1 failure
- **Tier 3**: OpenRouter (Emergency) - Last resort with multiple model options

**Smart Features:**

- **Rate limit detection** with exponential backoff
- **Health monitoring** of each tier in real-time
- **Zero-downtime switching** between providers
- **Request deduplication** to prevent redundant API calls
- **Cost optimization** through intelligent provider selection

#### Error Recovery

- Automatically fixes common SQL errors
- Retries failed operations
- Provides helpful error messages

---

## System Architecture

### High-Level Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Orchestrator   │───▶│  Query Engine   │
│  (Natural Lang) │    │ (Context + AI)  │    │  (SQL/Code/RAG) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │ Session Memory  │    │   AI Models     │
│  (Modern Chat)  │    │  (Persistence)  │    │ (GPT-4/Gemini)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Detailed Architecture Flow

```
User Query → Orchestrator → Query Classification → Handler Selection
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        │                                 │                                 │
                   ┌────▼────┐                     ┌──────▼──────┐                   ┌─────▼─────┐
                   │ SQL Mode │                     │  CODE Mode  │                   │ RAG Mode  │
                   └────┬────┘                     └──────┬──────┘                   └─────┬─────┘
                        │                                 │                                 │
                   ┌────▼────┐                     ┌──────▼──────┐                   ┌─────▼─────┐
                   │generate_│                     │ generate_   │                   │Semantic   │
                   │  sql()  │                     │   code()    │                   │ Search    │
                   └────┬────┘                     └──────┬──────┘                   └─────┬─────┘
                        │                                 │                                 │
                   ┌────▼────┐                     ┌──────▼──────┐                   ┌─────▼─────┐
                   │Self-    │                     │Self-         │                   │ChromaDB   │
                   │Validator│                     │Validator     │                   │Vectors    │
                   └────┬────┘                     └──────┬──────┘                   └─────┬─────┘
                        │                                 │                                 │
                        └─────────────────────────────────┼─────────────────────────────────┘
                                                          │
                                                     ┌────▼────┐
                                                     │Multi-Tier│
                                                     │AI Fallback│
                                                     │(OpenAI → │
                                                     │ Gemini → │
                                                     │OpenRouter)│
                                                     └───────────┘
```

### Component Breakdown

#### Frontend Layer (Streamlit)

- **Modern chat interface** with real-time messaging
- **Rich result display** (tables, charts, insights)
- **Query history** and session management

#### Orchestrator Layer

- **LLM-based query classification**: Intelligent routing to SQL/RAG/CODE handlers
- **Context management** for follow-up questions
- **Multi-tier AI coordination** with automatic fallback
- **6 action types**: SQL, RAG, CODE, SQL+RAG, CONVERSATION, METADATA

#### Backend Engine Layer

- **Query execution** across different modes
- **Database operations** with validation
- **Code generation** and execution with self-correction
- **Result formatting** and insights generation

#### Data Layer

- **SQLite database** with optimized schema
- **Vector embeddings** for semantic search (ChromaDB)
- **Session persistence** and caching
- **Generated content** storage

### Database Schema

**Core Tables:**

- `orders` - Order transactions with timestamps and status
- `customers` - Customer profiles with geographic data
- `order_items` - Product-quantity relationships
- `products` - Product catalog with categories
- `sellers` - Seller information and locations
- `order_payments` - Payment transactions and methods
- `order_reviews` - Customer feedback and ratings
- `category_translation` - Portuguese to English category names

**Key Relationships:**

- Orders → Customers (geographic analysis)
- Orders → Order Items → Products (product performance)
- Orders → Order Payments (revenue analysis)
- Orders → Order Reviews (satisfaction metrics)

### Data Flow

1. **User Input** → Natural language query
2. **Context Enhancement** → Add conversation history
3. **Intent Analysis** → Determine analysis type (SQL/Code/RAG)
4. **AI Processing** → Generate appropriate code/query
5. **Execution** → Run against database or execute code
6. **Result Formatting** → Create human-readable output
7. **Memory Update** → Store for future context

---

## Business Value & Advantages

### For Business Users

- **Self-service analytics**: No more waiting for IT reports
- **Faster decision making**: Get answers instantly
- **Reduced IT workload**: Business users handle their own questions
- **Better insights**: AI finds patterns humans might miss

### For Technical Teams

- **Faster development**: AI generates code and queries
- **Reduced maintenance**: Self-healing error recovery
- **Scalable architecture**: Handles multiple users simultaneously
- **Future-proof**: Easy to add new data sources and AI models

### Key Advantages

| Traditional Approach | CommServe Approach               |
| -------------------- | -------------------------------- |
| Write SQL queries    | Ask in plain English             |
| Wait for IT team     | Get instant answers              |
| Static reports       | Dynamic, conversational analysis |
| Manual data prep     | Automated insights               |
| Single data views    | Multi-dimensional analysis       |

---

## Technical Implementation

### Core Technologies

- **Frontend**: Streamlit (Python web framework)
- **Backend**: FastAPI (high-performance API)
- **Database**: SQLite (embedded, production-ready)
- **AI Models**: OpenAI GPT-4, Google Gemini, OpenRouter
- **Embeddings**: Sentence Transformers + ChromaDB
- **Visualization**: Matplotlib, Seaborn, Plotly

### Key Algorithms

#### Intelligent Query Classification

**LLM-powered routing** replaces hardcoded keyword matching:

```python
# Dynamic query understanding with context
query_types = {
    "SQL": "Database queries (sales, orders, customers)",
    "CODE": "Advanced analysis with visualizations",
    "RAG": "Business knowledge and documentation search",
    "METADATA": "Dataset overview and statistics",
    "CONVERSATION": "Chat and clarification",
    "HYBRID": "Combined SQL + RAG analysis"
}
```

**Features:**

- **Context awareness**: Follow-up questions like "what about last month?"
- **Confidence scoring**: Each classification includes reliability score
- **Fallback safety**: Keyword-based backup if LLM unavailable

#### Self-Validating Code Generation

**AI-driven code creation** with automatic error correction:

```python
# generate_code() workflow:
1. LLM generates Python code with database queries
2. Syntax validation and auto-fix
3. Execution with error handling
4. Result formatting and visualization
```

**Smart Features:**

- **Schema awareness**: Uses exact table/column names from database
- **Error recovery**: Automatically fixes common SQL and Python syntax errors
- **Visualization generation**: Creates matplotlib/seaborn charts automatically
- **Business insights**: Calculates and displays actionable metrics

#### Context-Aware Conversations

- Maintains conversation state across queries
- Automatically detects follow-up questions
- Provides contextual responses based on history

#### Self-Validating SQL Generation

- Generates SQL using advanced prompts
- Validates syntax before execution
- Auto-corrects common errors
- Provides execution statistics

### Security & Reliability

- **API Key Rotation**: Automatic fallback across providers
- **Rate Limit Handling**: Intelligent cooldown and retry logic
- **Input Validation**: Sanitizes all user inputs
- **Error Isolation**: Failures don't crash the entire system
- **Session Isolation**: User conversations are completely separate

---

## What's Currently Implemented

### Core Features

- [x] Natural language to SQL conversion
- [x] Python code generation and execution
- [x] Multi-tier AI fallback system
- [x] Conversation memory and context
- [x] Modern chat interface
- [x] Real-time result visualization
- [x] Multi-language support
- [x] Error recovery and validation

### Advanced Features

- [x] Sentiment analysis of reviews
- [x] Automated data insights
- [x] Session management

### Data Sources

- [x] Brazilian E-Commerce dataset (100k+ orders)
- [x] Customer reviews and ratings
- [x] Product catalog with categories
- [x] Seller performance metrics
- [x] Geographic data and trends

---

## Future Roadmap & Enhancements

- **Enhanced NLP**: Better understanding of complex business queries
- **Custom Data Connectors**: Support for PostgreSQL, MySQL, and other databases
- **Real-time Dashboards**: Live data monitoring and streaming updates
- **Advanced Visualizations**: Interactive dashboards and drill-down capabilities
- **Automated Reporting**: Scheduled report generation and distribution
- **Recommendation Engine**: AI-powered business insights and suggestions

---

## 📊 System Workflow

### 1. User Interaction

```
User types: "How are sales trending this quarter?"
           ↓
   Frontend captures input
           ↓
   Sends to backend API
```

### 2. Query Processing

```
Backend receives query
           ↓
   Orchestrator adds context
           ↓
   AI analyzes intent (→ SQL Mode)
           ↓
   Generates appropriate query
```

### 3. Execution & Results

```
Query executes against database
           ↓
   Results formatted for display
           ↓
   AI generates insights
           ↓
   Frontend displays rich output
```

### 4. Memory & Learning

```
Conversation stored in memory
           ↓
   Context available for follow-ups
           ↓
   System learns from interactions
```

---

## Testing & Quality Assurance

### Test Organization

The project uses a clean, organized testing structure:

- **`tests/`**: Core test suite with consolidated integration tests
- **`tests/test_utilities/`**: Development utilities and debug scripts
- **`tests/pyproject.toml`**: pytest configuration with coverage settings

### Automated Testing

**Important**: All tests must be run within the activated virtual environment to ensure dependencies are available.

**Note**: Coverage reporting requires `pytest-cov` (included in requirements.txt).

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows

# Run comprehensive test suite (34 tests total)
python -m pytest tests/ -v

# Run integration tests only
python -m pytest tests/ -k integration -v

# Run with coverage report
python -m pytest tests/ --cov=backend --cov-report=html

# View coverage report in browser
start htmlcov/index.html  # Windows
# or open htmlcov/index.html  # Linux/Mac

# Run specific test file
python -m pytest tests/test_integration.py -v

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest tests/ -n auto
```

### What the Tests Cover

The test suite validates all core functionality:

- ** AI Features**: Natural language processing, SQL generation, code generation, multilingual support
- ** Conversation Management**: Context awareness, follow-up questions, session memory
- ** System Integration**: Multi-tier AI fallback, rate limiting, error recovery
- ** Data Processing**: Database operations, result formatting, data validation
- ** Frontend**: UI components, configuration, imports
- ** Performance**: Response times, memory management, concurrent operations

**Test Statistics**: 34 tests covering 100% of implemented features (33 passed, 1 skipped when LLM providers unavailable).

### Troubleshooting Tests

**Issue**: `ModuleNotFoundError` for dependencies like `openai`, `langdetect`
**Solution**: Ensure you're running tests in the activated virtual environment:

```bash
.venv\Scripts\activate
python -m pytest tests/
```

**Issue**: Tests taking too long (>5 minutes)
**Solution**: Some AI-powered tests require API calls. Check your internet connection and API keys.

**Issue**: Some tests are skipped
**Solution**: Tests automatically skip when LLM providers (OpenAI/Gemini) are unavailable due to rate limits or API issues. This is normal behavior.

**Issue**: Database connection errors
**Solution**: Ensure the SQLite database is properly initialized by running the application first.

**Issue**: Import errors for local modules
**Solution**: Run tests from the project root directory where `tests/` is located.

### Test Structure

#### Core Tests (`tests/`)

- **`test_integration.py`**: Comprehensive integration testing covering:
  - Context and memory management
  - Conversational flow validation
  - System integration testing
  - Data processing pipelines

#### Test Utilities (`tests/test_utilities/`)

- **`analyze_reviews.py`**: Review data analysis utilities
- **`check_db.py`**: Database connection and schema validation
- **`checkemb.py`**: Embedding system diagnostics
- **`test_*.py`**: Various testing utilities and debug scripts

### Manual Testing Scenarios

- **Basic Queries**: Simple data retrieval and validation
- **Complex Analysis**: Multi-table joins and aggregations
- **Follow-up Questions**: Context maintenance and conversation flow
- **Error Recovery**: Invalid inputs and API failures
- **Performance**: Load testing with multiple concurrent users
- **Data Integrity**: Validation of processed datasets and embeddings

### Quality Metrics

- **Response Time**: Typically < 15 seconds for most queries (varies with AI provider load)
- **Query Understanding**: Intelligent LLM-based routing with confidence scoring
- **Reliability**: Multi-tier failover system ensures continued operation during API rate limits

---

### Architecture Guidelines

- **Modular Design**: Clear separation of concerns
- **Dependency Injection**: Loose coupling between components
- **Error Handling**: Graceful degradation, not crashes

---

_Thank you_
