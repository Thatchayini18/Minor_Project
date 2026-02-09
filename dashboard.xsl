<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:output method="html" indent="yes"/>

<!-- Paramètre pour la langue -->
<xsl:param name="lang" select="'en'"/>


<!-- Charger les traductions -->
<xsl:variable name="translations" select="document('translations.xml')"/>

<!-- Template de traduction -->
<xsl:template name="t">
    <xsl:param name="key"/>
    <xsl:value-of select="$translations/translations/translation[@key=$key]/lang[@code=$lang]"/>
</xsl:template>

<xsl:template match="/">
<html lang="{$lang}">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>NeoCity 2.0 | <xsl:call-template name="t"><xsl:with-param name="key" select="'dashboard_title'"/></xsl:call-template></title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"/>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/animejs@3.2.1/lib/anime.min.js"></script> 
    <link rel="stylesheet" href="./style.css" type="text/css"></link>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="header">
            <div class="city-info">
                <div class="city-logo">
                    <i class="fas fa-city"></i>
                </div>
                <div class="city-text">
                    <h1><xsl:value-of select="/smartCity/@name"/></h1>
                    <div class="city-subtitle">
                        <xsl:call-template name="t">
                            <xsl:with-param name="key" select="'dashboard_title'"/>
                        </xsl:call-template>
                    </div>
                </div>
            </div>
            
            <div class="city-stats">
                <div class="stat-badge">
                    <div class="stat-value"><xsl:value-of select="format-number(/smartCity/@population, '#,##0')"/></div>
                    <div class="stat-label">
                        <xsl:call-template name="t">
                            <xsl:with-param name="key" select="'population'"/>
                        </xsl:call-template>
                    </div>
                </div>
                <div class="stat-badge">
                    <div class="stat-value">v<xsl:value-of select="/smartCity/@version"/></div>
                    <div class="stat-label">
                        <xsl:call-template name="t">
                            <xsl:with-param name="key" select="'version'"/>
                        </xsl:call-template>
                    </div>
                </div>
                <div class="stat-badge">
                    <div class="stat-value">
                        <xsl:value-of select="/smartCity/cityInfo/status"/>
                    </div>
                    <div class="stat-label">
                        <xsl:call-template name="t">
                            <xsl:with-param name="key" select="'status'"/>
                        </xsl:call-template>
                    </div>
                </div>
            </div>
            
            <!-- Language Selector -->
            <div class="language-selector">
                <a href="?lang=en" class="lang-btn {$lang = 'en'}">
                    <i class="fas fa-globe"></i> EN
                </a>
                <a href="?lang=fr" class="lang-btn {$lang = 'fr'}">
                    <i class="fas fa-globe"></i> FR
                </a>
            </div>

<div class="logout-btn" onclick="logout()">
    <i class="fas fa-sign-out-alt"></i>
    <xsl:call-template name="t">
        <xsl:with-param name="key" select="'logout'"/>
    </xsl:call-template>
</div>

        </header>

        <!-- Navigation -->
        <nav class="nav-container">
            <div class="nav-scroll">
                <button class="nav-btn active" onclick="showSection('overview')">
                    <i class="fas fa-tachometer-alt"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'overview'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('traffic')">
                    <i class="fas fa-traffic-light"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'traffic'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('security')">
                    <i class="fas fa-shield-alt"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'security'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('environment')">
                    <i class="fas fa-leaf"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'environment'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('services')">
                    <i class="fas fa-hospital"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'services'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('iot')">
                    <i class="fas fa-microchip"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'iot'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('users')">
                    <i class="fas fa-users"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'users'"/>
                    </xsl:call-template>
                </button>
                <button class="nav-btn" onclick="showSection('analytics')">
                    <i class="fas fa-chart-line"></i>
                    <xsl:call-template name="t">
                        <xsl:with-param name="key" select="'analytics'"/>
                    </xsl:call-template>
                </button>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="main-content">
            
            <!-- Overview Section -->
            <section id="overview" class="section active">
                <!-- Quick Stats Row -->
                <div class="quick-stats" style="grid-column: span 12;">
                    <div class="quick-stat">
                        <div class="quick-stat-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'active_incidents'"/>
                            </xsl:call-template>
                        </div>
                        <div class="quick-stat-value pulse" style="color: var(--danger);">
                            <xsl:value-of select="count(/smartCity/security/incident[@status='Active' or @status='Investigating'])"/>
                        </div>
                    </div>
                    <div class="quick-stat">
                        <div class="quick-stat-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'iot_connected'"/>
                            </xsl:call-template>
                        </div>
                        <div class="quick-stat-value" style="color: var(--success);">
                            <xsl:value-of select="format-number(/smartCity/iotDevices/@connected, '#,##0')"/>
                        </div>
                    </div>
                    <div class="quick-stat">
                        <div class="quick-stat-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'air_quality'"/>
                            </xsl:call-template>
                        </div>
                        <div class="quick-stat-value">
                            <xsl:value-of select="count(/smartCity/environment/airQuality/sensor[qualityIndex='Good' or qualityIndex='Excellent'])"/>/<xsl:value-of select="count(/smartCity/environment/airQuality/sensor)"/>
                        </div>
                    </div>
                    <div class="quick-stat">
                        <div class="quick-stat-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'response_time'"/>
                            </xsl:call-template>
                        </div>
                        <div class="quick-stat-value">
                            <xsl:variable name="avgResponse" select="sum(/smartCity/security/incident/responseTime) div count(/smartCity/security/incident[responseTime])"/>
                            <xsl:choose>
                                <xsl:when test="$avgResponse &gt; 0">
                                    <xsl:value-of select="format-number($avgResponse, '0.0')"/>min
                                </xsl:when>
                                <xsl:otherwise>N/A</xsl:otherwise>
                            </xsl:choose>
                        </div>
                    </div>
                </div>

                <!-- Main Cards -->
                <div class="overview-grid">
                    <!-- City Health Score -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-heartbeat"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'city_health_score'"/>
                                </xsl:call-template>
                            </h3>
                            <span class="status-badge status-high">
                                <span class="status-dot"></span>
                                <xsl:value-of select="/smartCity/analytics/cityHealth/@trend"/>
                            </span>
                        </div>
                        <div class="kpi-value">
                            <xsl:value-of select="/smartCity/analytics/cityHealth/@score"/>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {/smartCity/analytics/cityHealth/@score}%"></div>
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-arrow-up trend-down"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'requires_attention'"/>
                                </xsl:call-template>
                            </span>
                        </div>
                    </div>

                    <!-- IoT Connectivity -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-wifi"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'iot_connectivity'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <xsl:variable name="iotPercent" select="round(/smartCity/iotDevices/@connected div /smartCity/iotDevices/@total * 100)"/>
                        <div class="kpi-value">
                            <xsl:value-of select="$iotPercent"/>%
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {$iotPercent}%"></div>
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-arrow-up trend-up"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'excellent_connection'"/>
                                </xsl:call-template>
                            </span>
                        </div>
                    </div>

                    <!-- Environment Status -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-temperature-high"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'environment_status'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="kpi-value">
                            <xsl:value-of select="/smartCity/environment/weather/current/temperature"/>°C
                        </div>
                        <div class="kpi-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'temperature'"/>
                            </xsl:call-template> • 
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'humidity'"/>
                            </xsl:call-template>: <xsl:value-of select="/smartCity/environment/weather/current/humidity"/>%
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-wind"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'wind'"/>
                                </xsl:call-template>: <xsl:value-of select="/smartCity/environment/weather/current/windSpeed"/> km/h
                            </span>
                        </div>
                    </div>

                    <!-- Recent Incidents Table -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-history"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'recent_incidents'"/>
                                </xsl:call-template>
                            </h3>
                            <button class="status-badge status-medium" onclick="showSection('security')" style="cursor: pointer;">
                                <span class="status-dot"></span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'view_all'"/>
                                </xsl:call-template>
                            </button>
                        </div>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'type'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'severity'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'zone'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'status_label'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'time'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'response'"/>
                                        </xsl:call-template>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                <xsl:for-each select="/smartCity/security/incident">
                                    <xsl:sort select="time" order="descending"/>
                                    <xsl:if test="position() &lt;= 5">
                                        <tr>
                                            <td><xsl:value-of select="@type"/></td>
                                            <td>
                                                <span class="status-badge status-{translate(@severity,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')}">
                                                    <span class="status-dot"></span>
                                                    <xsl:value-of select="@severity"/>
                                                </span>
                                            </td>
                                            <td><xsl:value-of select="zone"/></td>
                                            <td><xsl:value-of select="@status"/></td>
                                            <td><xsl:value-of select="substring(time, 12, 5)"/></td>
                                            <td>
                                                <xsl:choose>
                                                    <xsl:when test="responseTime">
                                                        <span style="color: var(--success); font-weight: 600;">
                                                            <xsl:value-of select="responseTime"/> min
                                                        </span>
                                                    </xsl:when>
                                                    <xsl:otherwise>
                                                        <span style="color: var(--warning);">
                                                            <xsl:call-template name="t">
                                                                <xsl:with-param name="key" select="'pending'"/>
                                                            </xsl:call-template>
                                                        </span>
                                                    </xsl:otherwise>
                                                </xsl:choose>
                                            </td>
                                        </tr>
                                    </xsl:if>
                                </xsl:for-each>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- Traffic Section -->
            <section id="traffic" class="section">
                <div class="overview-grid">
                    <!-- Traffic Overview -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-traffic-light"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'traffic_management'"/>
                                </xsl:call-template>
                            </h3>
                            <div class="quick-stats" style="display: flex; gap: 15px; margin: 0;">
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'total_intersections'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem;">
                                        <xsl:value-of select="count(/smartCity/infrastructure/traffic/intersection)"/>
                                    </div>
                                </div>
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'average_speed'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem;">
                                        <xsl:value-of select="format-number(sum(/smartCity/infrastructure/traffic/intersection/avgSpeed) div count(/smartCity/infrastructure/traffic/intersection), '0')"/> km/h
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Intersections Table -->
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Intersection ID</th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'zone'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'congestion_level'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'average_speed'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'last_update'"/>
                                        </xsl:call-template>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                <xsl:for-each select="/smartCity/infrastructure/traffic/intersection">
                                    <tr>
                                        <td><xsl:value-of select="@id"/></td>
                                        <td><xsl:value-of select="@zone"/></td>
                                        <td>
                                            <xsl:choose>
                                                <xsl:when test="congestionLevel='High' or congestionLevel='Severe'">
                                                    <span class="status-badge status-high">
                                                        <span class="status-dot"></span>
                                                        <xsl:value-of select="congestionLevel"/>
                                                    </span>
                                                </xsl:when>
                                                <xsl:when test="congestionLevel='Medium'">
                                                    <span class="status-badge status-medium">
                                                        <span class="status-dot"></span>
                                                        <xsl:value-of select="congestionLevel"/>
                                                    </span>
                                                </xsl:when>
                                                <xsl:otherwise>
                                                    <span class="status-badge status-low">
                                                        <span class="status-dot"></span>
                                                        <xsl:value-of select="congestionLevel"/>
                                                    </span>
                                                </xsl:otherwise>
                                            </xsl:choose>
                                        </td>
                                        <td><xsl:value-of select="avgSpeed"/> km/h</td>
                                        <td><xsl:value-of select="substring(lastUpdate, 12, 5)"/></td>
                                    </tr>
                                </xsl:for-each>
                            </tbody>
                        </table>
                    </div>

                    <!-- Public Transport -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-bus"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'public_transport'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="metrics-grid">
                            <xsl:for-each select="/smartCity/infrastructure/publicTransport/*">
                                <div class="kpi-card">
                                    <div class="kpi-value">
                                        <xsl:value-of select="@id"/>
                                    </div>
                                    <div class="kpi-label">
                                        <xsl:value-of select="local-name()"/>
                                    </div>
                                    <div class="kpi-trend">
                                        <xsl:choose>
                                            <xsl:when test="@status='Delayed'">
                                                <i class="fas fa-exclamation-triangle" style="color: var(--warning);"></i>
                                                <span>
                                                    <xsl:call-template name="t">
                                                        <xsl:with-param name="key" select="'delayed'"/>
                                                    </xsl:call-template>: <xsl:value-of select="delayMinutes"/> min
                                                </span>
                                            </xsl:when>
                                            <xsl:otherwise>
                                                <i class="fas fa-check-circle" style="color: var(--success);"></i>
                                                <span>
                                                    <xsl:call-template name="t">
                                                        <xsl:with-param name="key" select="'on_time'"/>
                                                    </xsl:call-template>: <xsl:value-of select="delayMinutes"/> min
                                                </span>
                                            </xsl:otherwise>
                                        </xsl:choose>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.85rem;">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'passengers'"/>
                                        </xsl:call-template>: <xsl:value-of select="passengerCount"/>
                                    </div>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>

                    <!-- Energy Management -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-bolt"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'energy_status'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="kpi-value" style="font-size: 2rem;">
                            <xsl:value-of select="/smartCity/infrastructure/energy/smartGrid/@loadPercentage"/>%
                        </div>
                        <div class="kpi-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'smart_grid_load'"/>
                            </xsl:call-template>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {/smartCity/infrastructure/energy/smartGrid/@loadPercentage}%"></div>
                        </div>
                        <div style="margin-top: 20px;">
                            <h4>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'power_plants'"/>
                                </xsl:call-template>
                            </h4>
                            <xsl:for-each select="/smartCity/infrastructure/energy/powerPlant">
                                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                                    <strong><xsl:value-of select="@id"/></strong>
                                    (<xsl:value-of select="@type"/>)
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'output'"/>
                                    </xsl:call-template>: <xsl:value-of select="@outputMW"/> MW
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'efficiency'"/>
                                    </xsl:call-template>: <xsl:value-of select="efficiency"/>%
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Security Section -->
            <section id="security" class="section">
                <div class="overview-grid">
                    <!-- Security Overview -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-shield-alt"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'security_incidents'"/>
                                </xsl:call-template>
                            </h3>
                            <div class="quick-stats" style="display: flex; gap: 15px; margin: 0;">
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'total_incidents'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem;">
                                        <xsl:value-of select="count(/smartCity/security/incident)"/>
                                    </div>
                                </div>
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'active'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem; color: var(--danger);">
                                        <xsl:value-of select="count(/smartCity/security/incident[@status='Active' or @status='Investigating'])"/>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Incidents Table -->
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Incident ID</th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'type'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'severity'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'zone'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'status_label'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'time'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'assigned_to'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'response_time'"/>
                                        </xsl:call-template>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                <xsl:for-each select="/smartCity/security/incident">
                                    <xsl:sort select="time" order="descending"/>
                                    <tr>
                                        <td><xsl:value-of select="@id"/></td>
                                        <td><xsl:value-of select="@type"/></td>
                                        <td>
                                            <span class="status-badge status-{translate(@severity,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')}">
                                                <span class="status-dot"></span>
                                                <xsl:value-of select="@severity"/>
                                            </span>
                                        </td>
                                        <td><xsl:value-of select="zone"/></td>
                                        <td>
                                            <xsl:choose>
                                                <xsl:when test="@status='Active'">
                                                    <span class="status-badge status-high">
                                                        <span class="status-dot"></span>
                                                        <xsl:call-template name="t">
                                                            <xsl:with-param name="key" select="'active'"/>
                                                        </xsl:call-template>
                                                    </span>
                                                </xsl:when>
                                                <xsl:when test="@status='Investigating'">
                                                    <span class="status-badge status-medium">
                                                        <span class="status-dot"></span>
                                                        <xsl:call-template name="t">
                                                            <xsl:with-param name="key" select="'investigating'"/>
                                                        </xsl:call-template>
                                                    </span>
                                                </xsl:when>
                                                <xsl:when test="@status='Resolved'">
                                                    <span class="status-badge status-low">
                                                        <span class="status-dot"></span>
                                                        <xsl:call-template name="t">
                                                            <xsl:with-param name="key" select="'resolved'"/>
                                                        </xsl:call-template>
                                                    </span>
                                                </xsl:when>
                                                <xsl:when test="@status='Closed'">
                                                    <span class="status-badge status-low">
                                                        <span class="status-dot"></span>
                                                        <xsl:call-template name="t">
                                                            <xsl:with-param name="key" select="'closed'"/>
                                                        </xsl:call-template>
                                                    </span>
                                                </xsl:when>
                                                <xsl:otherwise>
                                                    <xsl:value-of select="@status"/>
                                                </xsl:otherwise>
                                            </xsl:choose>
                                        </td>
                                        <td><xsl:value-of select="substring(time, 12, 5)"/></td>
                                        <td><xsl:value-of select="assignedTo"/></td>
                                        <td>
                                            <xsl:choose>
                                                <xsl:when test="responseTime">
                                                    <span style="color: var(--success); font-weight: 600;">
                                                        <xsl:value-of select="responseTime"/> min
                                                    </span>
                                                </xsl:when>
                                                <xsl:otherwise>
                                                    <span style="color: var(--warning);">
                                                        <xsl:call-template name="t">
                                                            <xsl:with-param name="key" select="'pending'"/>
                                                        </xsl:call-template>
                                                    </span>
                                                </xsl:otherwise>
                                            </xsl:choose>
                                        </td>
                                    </tr>
                                </xsl:for-each>
                            </tbody>
                        </table>
                    </div>

                    <!-- Surveillance Cameras -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-video"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'surveillance_system'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="metrics-grid">
                            <xsl:for-each select="/smartCity/security/surveillance/camera">
                                <div class="kpi-card">
                                    <div class="kpi-value">
                                        <xsl:value-of select="@id"/>
                                    </div>
                                    <div class="kpi-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'zone'"/>
                                        </xsl:call-template>: <xsl:value-of select="@zone"/>
                                    </div>
                                    <div class="kpi-trend">
                                        <xsl:choose>
                                            <xsl:when test="@status='Active'">
                                                <i class="fas fa-check-circle" style="color: var(--success);"></i>
                                                <span>
                                                    <xsl:call-template name="t">
                                                        <xsl:with-param name="key" select="'active'"/>
                                                    </xsl:call-template>
                                                </span>
                                            </xsl:when>
                                            <xsl:otherwise>
                                                <i class="fas fa-tools" style="color: var(--warning);"></i>
                                                <span>Maintenance</span>
                                            </xsl:otherwise>
                                        </xsl:choose>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.85rem;">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'last_motion'"/>
                                        </xsl:call-template>: <xsl:value-of select="substring(lastMotion, 12, 5)"/>
                                        <br/>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'ai_detection'"/>
                                        </xsl:call-template>: <xsl:value-of select="aiDetection"/>
                                    </div>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Environment Section -->
            <section id="environment" class="section">
                <div class="overview-grid">
                    <!-- Air Quality -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-wind"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'air_quality_monitoring'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="metrics-grid">
                            <xsl:for-each select="/smartCity/environment/airQuality/sensor">
                                <div class="kpi-card">
                                    <div class="kpi-value">
                                        <xsl:value-of select="qualityIndex"/>
                                    </div>
                                    <div class="kpi-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'zone'"/>
                                        </xsl:call-template>: <xsl:value-of select="@zone"/>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {pm25 div 100 * 100}%"></div>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.8rem;">
                                        PM2.5: <xsl:value-of select="pm25"/>
                                        <br/>
                                        PM10: <xsl:value-of select="pm10"/>
                                        <br/>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'last_update'"/>
                                        </xsl:call-template>: <xsl:value-of select="substring(timestamp, 12, 5)"/>
                                    </div>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>

                    <!-- Weather -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-sun"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'current_weather'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="kpi-value">
                            <xsl:value-of select="/smartCity/environment/weather/current/temperature"/>°C
                        </div>
                        <div class="kpi-label">
                            <xsl:value-of select="/smartCity/environment/weather/current/condition"/>
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-tint"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'humidity'"/>
                                </xsl:call-template>: <xsl:value-of select="/smartCity/environment/weather/current/humidity"/>%
                            </span>
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-wind"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'wind'"/>
                                </xsl:call-template>: <xsl:value-of select="/smartCity/environment/weather/current/windSpeed"/> km/h
                            </span>
                        </div>
                        
                        <!-- Forecast -->
                        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                            <h4>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'tomorrow_forecast'"/>
                                </xsl:call-template>
                            </h4>
                            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                                <div>
                                    <div style="font-size: 1.2rem; font-weight: bold;">
                                        <xsl:value-of select="/smartCity/environment/weather/forecast[@period='Tomorrow']/high"/>°C
                                    </div>
                                    <div style="font-size: 0.8rem;">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'high'"/>
                                        </xsl:call-template>
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 1.2rem; font-weight: bold;">
                                        <xsl:value-of select="/smartCity/environment/weather/forecast[@period='Tomorrow']/low"/>°C
                                    </div>
                                    <div style="font-size: 0.8rem;">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'low'"/>
                                        </xsl:call-template>
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 1.2rem;">
                                        <xsl:value-of select="/smartCity/environment/weather/forecast[@period='Tomorrow']/condition"/>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Waste Management -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-trash-alt"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'waste_management'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="kpi-value">
                            <xsl:value-of select="/smartCity/environment/wasteManagement/recyclingRate"/>%
                        </div>
                        <div class="kpi-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'recycling_rate'"/>
                            </xsl:call-template>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {/smartCity/environment/wasteManagement/recyclingRate}%"></div>
                        </div>
                        
                        <!-- Smart Bins -->
                        <div style="margin-top: 20px;">
                            <h4>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'smart_bins'"/>
                                </xsl:call-template>
                            </h4>
                            <xsl:for-each select="/smartCity/environment/wasteManagement/smartBin">
                                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                                    <strong><xsl:value-of select="@id"/></strong> - <xsl:value-of select="@zone"/>
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'fill_level'"/>
                                    </xsl:call-template>: <xsl:value-of select="@fillLevel"/>%
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'next_collection'"/>
                                    </xsl:call-template>: <xsl:value-of select="substring(nextCollection, 12, 5)"/>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Services Section -->
            <section id="services" class="section">
                <div class="overview-grid">
                    <!-- Hospital -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-hospital"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'hospital_status'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <xsl:variable name="totalBeds" select="sum(/smartCity/services/hospital/@bedsAvailable)"/>
                        <xsl:variable name="totalHospitals" select="count(/smartCity/services/hospital)"/>
                        <div class="kpi-value">
                            <xsl:value-of select="$totalBeds"/>
                        </div>
                        <div class="kpi-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'total_beds_available'"/>
                            </xsl:call-template> (<xsl:value-of select="$totalHospitals"/>
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'hospitals'"/>
                            </xsl:call-template>)
                        </div>
                        <div class="kpi-trend">
                            <i class="fas fa-clock"></i>
                            <span>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'avg_waiting'"/>
                                </xsl:call-template>: <xsl:value-of select="format-number(sum(/smartCity/services/hospital/@waitingTime) div $totalHospitals, '0')"/> min
                            </span>
                        </div>
                        <div style="margin-top: 15px;">
                            <h4>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'all_hospitals'"/>
                                </xsl:call-template>
                            </h4>
                            <xsl:for-each select="/smartCity/services/hospital">
                                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                                    <strong><xsl:value-of select="@id"/></strong> - <xsl:value-of select="@zone"/>
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'beds'"/>
                                    </xsl:call-template>: <xsl:value-of select="@bedsAvailable"/> • 
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'wait'"/>
                                    </xsl:call-template>: <xsl:value-of select="@waitingTime"/> min
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>

                    <!-- School -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-school"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'school_status'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <xsl:variable name="totalStudents" select="sum(/smartCity/services/school/studentsPresent)"/>
                        <xsl:variable name="totalSchools" select="count(/smartCity/services/school)"/>
                        <div class="kpi-value">
                            <xsl:value-of select="$totalStudents"/>
                        </div>
                        <div class="kpi-label">
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'students_present'"/>
                            </xsl:call-template> (<xsl:value-of select="$totalSchools"/>
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'schools'"/>
                            </xsl:call-template>)
                        </div>
                        <div class="kpi-trend">
                            <xsl:variable name="openSchools" select="count(/smartCity/services/school[@status='Open'])"/>
                            <i class="fas fa-check-circle" style="color: var(--success);"></i>
                            <span><xsl:value-of select="$openSchools"/>/<xsl:value-of select="$totalSchools"/>
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'open'"/>
                            </xsl:call-template></span>
                        </div>
                        <div style="margin-top: 15px;">
                            <h4>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'all_schools'"/>
                                </xsl:call-template>
                            </h4>
                            <xsl:for-each select="/smartCity/services/school">
                                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                                    <strong><xsl:value-of select="@id"/></strong> - <xsl:value-of select="@zone"/>
                                    <br/>
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'status_label'"/>
                                    </xsl:call-template>: <xsl:value-of select="@status"/> • 
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'students'"/>
                                    </xsl:call-template>: <xsl:value-of select="studentsPresent"/>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>

                    <!-- Public WiFi -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-wifi"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'public_wifi'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <xsl:for-each select="/smartCity/services/publicWiFi">
                            <div class="kpi-value">
                                <xsl:value-of select="@uptime"/>%
                            </div>
                            <div class="kpi-label">
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'uptime'"/>
                                </xsl:call-template>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {@uptime}%"></div>
                            </div>
                            <div style="margin-top: 20px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">
                                            <xsl:value-of select="@accessPoints"/>
                                        </div>
                                        <div style="font-size: 0.8rem;">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'access_points'"/>
                                            </xsl:call-template>
                                        </div>
                                    </div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">
                                            <xsl:value-of select="activeConnections"/>
                                        </div>
                                        <div style="font-size: 0.8rem;">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'active_connections'"/>
                                            </xsl:call-template>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </xsl:for-each>
                    </div>
                </div>
            </section>

            <!-- IoT Section -->
            <section id="iot" class="section">
                <div class="overview-grid">
                    <!-- IoT Overview -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-microchip"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'iot_network_status'"/>
                                </xsl:call-template>
                            </h3>
                            <div class="quick-stats" style="display: flex; gap: 15px; margin: 0;">
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'total_devices'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem;">
                                        <xsl:value-of select="format-number(/smartCity/iotDevices/@total, '#,##0')"/>
                                    </div>
                                </div>
                                <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                    <div class="quick-stat-label">
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'connected'"/>
                                        </xsl:call-template>
                                    </div>
                                    <div class="quick-stat-value" style="font-size: 1.5rem; color: var(--success);">
                                        <xsl:value-of select="format-number(/smartCity/iotDevices/@connected, '#,##0')"/>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- IoT Devices Breakdown -->
                        <div class="metrics-grid" style="margin-top: 20px;">
                            <xsl:for-each select="/smartCity/iotDevices/device">
                                <div class="kpi-card">
                                    <div class="kpi-value">
                                        <xsl:value-of select="format-number(@count, '#,##0')"/>
                                    </div>
                                    <div class="kpi-label">
                                        <xsl:value-of select="@type"/>
                                    </div>
                                    <xsl:variable name="percent" select="round(@count div /smartCity/iotDevices/@total * 100)"/>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {$percent}%"></div>
                                    </div>
                                    <div class="kpi-trend">
                                        <span><xsl:value-of select="$percent"/>%
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'of_total'"/>
                                        </xsl:call-template></span>
                                    </div>
                                </div>
                            </xsl:for-each>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Users Section -->
            <section id="users" class="section">
                <div class="dashboard-card" style="grid-column: span 12;">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-users"></i>
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'system_users'"/>
                            </xsl:call-template>
                        </h3>
                        <div class="quick-stats" style="display: flex; gap: 15px; margin: 0;">
                            <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                <div class="quick-stat-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'total_users'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="quick-stat-value" style="font-size: 1.5rem;">
                                    <xsl:value-of select="count(/smartCity/users/user)"/>
                                </div>
                            </div>
                            <div class="quick-stat" style="padding: 10px 15px; min-width: auto;">
                                <div class="quick-stat-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'active_now'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="quick-stat-value" style="font-size: 1.5rem; color: var(--success);">
                                    <xsl:value-of select="count(/smartCity/users/user[@status='Active'])"/>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- User Cards Grid -->
                    <div class="user-grid">
                        <xsl:for-each select="/smartCity/users/user">
                            <xsl:sort select="name"/>
                            <div class="user-card">
                                <div class="user-header">
                                    <div class="user-avatar">
                                        <xsl:value-of select="substring(name, 1, 1)"/>
                                    </div>
                                    <div class="user-info">
                                        <h3><xsl:value-of select="name"/></h3>
                                        <div class="user-role">
                                            <span class="role-badge role-admin">
                                                <i class="fas fa-user"></i>
                                                <xsl:value-of select="@department"/>
                                            </span>
                                            <span class="status-badge status-{translate(@status,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')}">
                                                <span class="status-dot"></span>
                                                <xsl:value-of select="@status"/>
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="user-details">
                                    <div class="user-detail">
                                        <span class="detail-label">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'user_id'"/>
                                            </xsl:call-template>:
                                        </span>
                                        <span class="detail-value"><xsl:value-of select="@id"/></span>
                                    </div>
                                    <div class="user-detail">
                                        <span class="detail-label">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'email'"/>
                                            </xsl:call-template>:
                                        </span>
                                        <span class="detail-value"><xsl:value-of select="email"/></span>
                                    </div>
                                    <div class="user-detail">
                                        <span class="detail-label">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'department'"/>
                                            </xsl:call-template>:
                                        </span>
                                        <span class="detail-value"><xsl:value-of select="@department"/></span>
                                    </div>
                                    <div class="user-detail">
                                        <span class="detail-label">
                                            <xsl:call-template name="t">
                                                <xsl:with-param name="key" select="'last_activity'"/>
                                            </xsl:call-template>:
                                        </span>
                                        <span class="detail-value">
                                            <xsl:value-of select="substring(lastLogin, 12, 5)"/>
                                            <xsl:text> today</xsl:text>
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </xsl:for-each>
                    </div>
                    
                    <!-- User Summary Table -->
                    <div style="margin-top: 30px;">
                        <h3 class="card-title" style="margin-bottom: 15px;">
                            <i class="fas fa-table"></i>
                            <xsl:call-template name="t">
                                <xsl:with-param name="key" select="'user_details_table'"/>
                            </xsl:call-template>
                        </h3>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'user_id'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'name'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'status_label'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'department'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'email'"/>
                                        </xsl:call-template>
                                    </th>
                                    <th>
                                        <xsl:call-template name="t">
                                            <xsl:with-param name="key" select="'last_activity'"/>
                                        </xsl:call-template>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                <xsl:for-each select="/smartCity/users/user">
                                    <xsl:sort select="name"/>
                                    <tr>
                                        <td><xsl:value-of select="@id"/></td>
                                        <td><strong><xsl:value-of select="name"/></strong></td>
                                        <td>
                                            <span class="status-badge status-{translate(@status,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')}">
                                                <span class="status-dot"></span>
                                                <xsl:value-of select="@status"/>
                                            </span>
                                        </td>
                                        <td><xsl:value-of select="@department"/></td>
                                        <td><xsl:value-of select="email"/></td>
                                        <td>
                                            <xsl:value-of select="substring(lastLogin, 12, 5)"/>
                                        </td>
                                    </tr>
                                </xsl:for-each>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- Analytics Section -->
            <section id="analytics" class="section">
                <div class="overview-grid">
                    <!-- City Analytics Overview -->
                    <div class="dashboard-card" style="grid-column: span 12;">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-chart-line"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'city_analytics_dashboard'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        
                        <!-- Analytics KPIs -->
                        <div class="metrics-grid">
                            <div class="kpi-card">
                                <div class="kpi-value">
                                    <xsl:value-of select="/smartCity/analytics/cityHealth/@score"/>
                                </div>
                                <div class="kpi-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'city_health_score'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {/smartCity/analytics/cityHealth/@score}%"></div>
                                </div>
                                <div class="kpi-trend">
                                    <xsl:choose>
                                        <xsl:when test="/smartCity/analytics/cityHealth/@trend='Improving'">
                                            <i class="fas fa-arrow-up trend-up"></i>
                                            <span>Trend: <xsl:value-of select="/smartCity/analytics/cityHealth/@trend"/></span>
                                        </xsl:when>
                                        <xsl:otherwise>
                                            <i class="fas fa-arrow-down trend-down"></i>
                                            <span>Trend: <xsl:value-of select="/smartCity/analytics/cityHealth/@trend"/></span>
                                        </xsl:otherwise>
                                    </xsl:choose>
                                </div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-value">
                                    <xsl:value-of select="/smartCity/analytics/safetyIndex/@score"/>
                                </div>
                                <div class="kpi-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'safety_index'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {/smartCity/analytics/safetyIndex/@score}%"></div>
                                </div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-value">
                                    <xsl:value-of select="/smartCity/analytics/sustainabilityIndex/@score"/>
                                </div>
                                <div class="kpi-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'sustainability_index'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {/smartCity/analytics/sustainabilityIndex/@score}%"></div>
                                </div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-value">
                                    <xsl:value-of select="/smartCity/analytics/citizenSatisfaction/@score"/>
                                </div>
                                <div class="kpi-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'citizen_satisfaction'"/>
                                    </xsl:call-template>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {/smartCity/analytics/citizenSatisfaction/@score}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- City Information -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-info-circle"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'city_information'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="user-details">
                            <xsl:for-each select="/smartCity/cityInfo/*">
                                <div class="user-detail">
                                    <span class="detail-label">
                                        <xsl:choose>
                                            <xsl:when test="name()='mayor'">Mayor</xsl:when>
                                            <xsl:when test="name()='emergencyNumber'">
                                                <xsl:call-template name="t">
                                                    <xsl:with-param name="key" select="'emergency'"/>
                                                </xsl:call-template>
                                            </xsl:when>
                                            <xsl:when test="name()='website'">
                                                <xsl:call-template name="t">
                                                    <xsl:with-param name="key" select="'website'"/>
                                                </xsl:call-template>
                                            </xsl:when>
                                            <xsl:when test="name()='timezone'">
                                                <xsl:call-template name="t">
                                                    <xsl:with-param name="key" select="'timezone'"/>
                                                </xsl:call-template>
                                            </xsl:when>
                                            <xsl:when test="name()='status'">
                                                <xsl:call-template name="t">
                                                    <xsl:with-param name="key" select="'status'"/>
                                                </xsl:call-template>
                                            </xsl:when>
                                        </xsl:choose>
                                    </span>
                                    <span class="detail-value">
                                        <xsl:choose>
                                            <xsl:when test="name()='website'">
                                                <a href="{.}" style="color: var(--primary-light); text-decoration: none;">
                                                    <xsl:value-of select="."/>
                                                </a>
                                            </xsl:when>
                                            <xsl:otherwise>
                                                <xsl:value-of select="."/>
                                            </xsl:otherwise>
                                        </xsl:choose>
                                    </span>
                                </div>
                            </xsl:for-each>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'population'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="format-number(/smartCity/@population, '#,##0')"/></span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'version'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="/smartCity/@version"/></span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'date'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="/smartCity/@date"/></span>
                            </div>
                        </div>
                    </div>

                    <!-- City Summary -->
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-chart-pie"></i>
                                <xsl:call-template name="t">
                                    <xsl:with-param name="key" select="'city_summary'"/>
                                </xsl:call-template>
                            </h3>
                        </div>
                        <div class="user-details">
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'total_incidents_today'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="count(/smartCity/security/incident)"/></span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'iot_connectivity'"/>
                                    </xsl:call-template>
                                </span>
                                <xsl:variable name="iotPercent" select="round(/smartCity/iotDevices/@connected div /smartCity/iotDevices/@total * 100)"/>
                                <span class="detail-value"><xsl:value-of select="$iotPercent"/>%</span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'public_transport_status'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value">
                                    <xsl:value-of select="count(/smartCity/infrastructure/publicTransport/*[@status='Operational'])"/>/<xsl:value-of select="count(/smartCity/infrastructure/publicTransport/*)"/>
                                </span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'air_quality_sensors'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="count(/smartCity/environment/airQuality/sensor)"/></span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'active_cameras'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value">
                                    <xsl:value-of select="count(/smartCity/security/surveillance/camera[@status='Active'])"/>/<xsl:value-of select="count(/smartCity/security/surveillance/camera)"/>
                                </span>
                            </div>
                            <div class="user-detail">
                                <span class="detail-label">
                                    <xsl:call-template name="t">
                                        <xsl:with-param name="key" select="'hospital_beds_available'"/>
                                    </xsl:call-template>
                                </span>
                                <span class="detail-value"><xsl:value-of select="sum(/smartCity/services/hospital/@bedsAvailable)"/></span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

        </main>
    </div>

    <script>
        // Logout function
function logout() {
    // Clear session storage
    sessionStorage.removeItem('neocity_authenticated');
    sessionStorage.removeItem('neocity_user');
    
    // Redirect to login page
    window.location.href = 'index.html';
}
        // Navigation
        function showSection(sectionId) {
            // Remove active class from all buttons and sections
            document.querySelectorAll('.nav-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });

            // Add active class to clicked button
            event.currentTarget.classList.add('active');
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
        }

        // Initialize animations when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Animate progress bars
            setTimeout(() => {
                document.querySelectorAll('.progress-fill').forEach(bar => {
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => {
                        bar.style.width = width;
                    }, 100);
                });
            }, 500);
            
            // Add hover effects
            document.querySelectorAll('.dashboard-card, .kpi-card, .user-card').forEach(card => {
                card.addEventListener('mouseenter', () => {
                    card.style.transform = 'translateY(-5px)';
                });
                
                card.addEventListener('mouseleave', () => {
                    card.style.transform = 'translateY(0)';
                });
            });
            
            // Update active language button
            const urlParams = new URLSearchParams(window.location.search);
            const lang = urlParams.get('lang') || 'en';
            document.querySelectorAll('.lang-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('href').includes(`lang=${lang}`)) {
                    btn.classList.add('active');
                }
            });
        });

        // Real-time update simulation
        setInterval(() => {
            // Animate pulse elements
            const pulseElements = document.querySelectorAll('.pulse');
            pulseElements.forEach(el => {
                el.style.opacity = '0.6';
                setTimeout(() => {
                    el.style.opacity = '1';
                }, 300);
            });
        }, 2000);
    </script>
    <script src="./dashboard.js"></script>
</body>
</html>
</xsl:template>
</xsl:stylesheet>
