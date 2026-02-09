🌆 EcoCity 2.0 – Smart City Dashboard


🌐 Live Demo



The project is deployed and fully functional. You can explore the live dashboard here:

🔗 https://github.com/Thatchayini18/Minor_Project.git

Test Credentials (from your smartcity.xml file):
- **Admin**: `thatchu@gmail.com` / `thatchu1234` (Full access)
- **Security Officer**: `iamrose@neocity.gov` / `Rose456!`
- **Environmental Officer**: `iamsaras@gmail.com` / `Saras@2005`



## 📋 Project Overview
EcoCity 2.0 is a comprehensive smart city monitoring and management system with a modern, responsive web interface. The system provides real-time insights into city infrastructure, security, environment, and public services through an interactive dashboard with role-based access control.

## 🏗️ System Architecture

### Core Components:
1. **Authentication System** - Secure login with XML-based user management
2. **XML Data Repository** - Centralized city data storage in `smartcity.xml`
3. **XSLT Dashboard Engine** - Dynamic HTML generation with multilingual support
4. **Role-Based Access Control** - Granular permission system
5. **Responsive Web Interface** - Modern UI with real-time monitoring

## 📁 Project Structure
EcoCity/
│
├── 📄 index.html # Login page with modern UI
├── 📄 login.js # Authentication logic
├── 📄 smartcity.xml # Main data repository
├── 📄 smartcity.xsd # XML schema validation
├── 📄 dashboard.xsl # XSLT transformation for dashboard
├── 📄 style.css # Dashboard styling
├── 📄 dashboard.js # Navigation and permissions logic
├── 📄 translations.xml # Multi-language support

## 🔐 Authentication System

### Features:
- **Secure Login**: Password protection with visibility toggle
- **XML User Database**: Users stored in `smartcity.xml` with hashed passwords
- **Session Management**: Uses `sessionStorage` for authentication state
- **Role-Based Permissions**: Four permission levels with section access control

### Permission Levels:
1. **`full`** - Access to all dashboard sections
2. **`monitor,alert,respond`** - Security-focused access
3. **`monitor,analyze`** - Traffic and analytics access
4. **`monitor,report`** - Environment monitoring access



## 📊 Dashboard Features

### 8 Main Sections:
1. **Overview** - City health score, active incidents, quick stats
2. **Traffic** - Intersection monitoring, public transport status, energy management
3. **Security** - Incident tracking, surveillance cameras, response times
4. **Environment** - Air quality sensors, weather data, waste management
5. **Services** - Hospital beds, school status, public WiFi
6. **IoT** - Device connectivity, network status
7. **Users** - System user management and monitoring
8. **Analytics** - City performance metrics and trends

### UI Features:
- **Modern Design**: Glass-morphism effects with gradient backgrounds
- **Responsive Layout**: Mobile-optimized interface
- **Real-time Updates**: Animated progress bars and live data
- **Interactive Elements**: Hover effects, animations, and transitions
- **Dark Theme**: Professional dashboard aesthetic

## 🌐 Multi-language Support

### Supported Languages:
- **English** (Default)
- **French** (Français)

### Implementation:
- Translations stored in `translations.xml`
- Language switching via URL parameter (`?lang=en` or `?lang=fr`)
- Dynamic text replacement using XSLT templates


## 🛡️ Security Features

### Data Protection:
- **XML Validation**: Strict schema validation via `smartcity.xsd`
- **Secure Credentials**: Passwords stored as attributes in XML
- **Session Management**: Authentication state in browser session
- **Access Control**: Section-level permission filtering

### Schema Validation:
- Data type validation (dates, numbers, enums)
- Required field enforcement
- Cross-referencing between elements (e.g., incident assignments)
## 🔄 Data Flow
Login → Authentication → Permission Check → Dashboard Generation
↓
XML Data → XSLT Transformation → HTML Dashboard
↓
User Interaction → JavaScript Updates → Real-time Display

## 🚀 Getting Started
### Quick Start:
1. **Start a local server** in the project directory:
   ```bash
   # Using Python
   python -m http.server 8000
2.Open browser and navigate to:
   ```bash
http://localhost:8000/index.html
```
3.Login using one of the sample accounts
4.Explore dashboard sections based on your permissions

Version: 2.0.0
Last Updated: January 2026




