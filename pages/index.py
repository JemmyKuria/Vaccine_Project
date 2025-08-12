import streamlit as st
from streamlit.components.v1 import html

# Your HTML content (paste your entire HTML code here)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VaxInsight - AI-Powered Vaccine Recommendations</title>
    <style>
        /* Global Styles */
        :root {
            --primary: #2563eb;
            --secondary: #1e40af;
            --accent: #3b82f6;
            --light: #f8fafc;
            --dark: #1e293b;
            --success: #10b981;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            line-height: 1.6;
            color: var(--dark);
            background-color: var(--light);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        section {
            padding: 80px 0;
        }
        
        h1, h2, h3 {
            margin-bottom: 20px;
            font-weight: 700;
        }
        
        h1 {
            font-size: 2.5rem;
        }
        
        h2 {
            font-size: 2rem;
            text-align: center;
            margin-bottom: 50px;
            position: relative;
        }
        
        h2::after {
            content: '';
            display: block;
            width: 80px;
            height: 4px;
            background: var(--primary);
            margin: 15px auto;
            border-radius: 2px;
        }
        
        p {
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .btn {
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 12px 30px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 2px solid var(--primary);
        }
        
        .btn:hover {
            background: transparent;
            color: var(--primary);
        }
        
        .btn-outline {
            background: transparent;
            color: var(--primary);
            border: 2px solid var(--primary);
        }
        
        .btn-outline:hover {
            background: var(--primary);
            color: white;
        }
        
        /* Header */
        header {
            background-color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: fixed;
            width: 100%;
            z-index: 100;
        }
        
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            text-decoration: none;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
        }
        
        .nav-links li {
            margin-left: 30px;
        }
        
        .nav-links a {
            text-decoration: none;
            color: var(--dark);
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .nav-links a:hover {
            color: var(--primary);
        }
        
        /* Hero Section */
        #hero {
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            padding: 150px 0 100px;
            text-align: center;
        }
        
        #hero h1 {
            font-size: 3rem;
            margin-bottom: 20px;
        }
        
        #hero p {
            max-width: 700px;
            margin: 0 auto 30px;
            font-size: 1.2rem;
        }
        
        /* About Section */
        #about {
            background-color: white;
        }
        
        .about-content {
            display: flex;
            align-items: center;
            gap: 50px;
        }
        
        .about-text {
            flex: 1;
        }
        
        .about-image {
            flex: 1;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .about-image img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        /* Features Section */
        #features {
            background-color: #f8fafc;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }
        
        .feature-card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            color: var(--primary);
            margin-bottom: 20px;
        }
        
        .feature-card h3 {
            font-size: 1.5rem;
        }
        
        /* How It Works */
        .steps {
            display: flex;
            flex-direction: column;
            gap: 40px;
            position: relative;
        }
        
        .step {
            display: flex;
            gap: 30px;
            position: relative;
            z-index: 1;
        }
        
        .step-number {
            background: var(--primary);
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            flex-shrink: 0;
        }
        
        .step-content {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            flex: 1;
        }
        
        /* CTA Section */
        #cta {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            text-align: center;
        }
        
        #cta h2 {
            color: white;
        }
        
        #cta h2::after {
            background: white;
        }
        
        #cta p {
            max-width: 700px;
            margin: 0 auto 30px;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        #cta .btn {
            background: white;
            color: var(--primary);
            border-color: white;
        }
        
        #cta .btn:hover {
            background: transparent;
            color: white;
        }
        
        /* Footer */
        footer {
            background: var(--dark);
            color: white;
            padding: 50px 0 20px;
        }
        
        .footer-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 40px;
            margin-bottom: 40px;
        }
        
        .footer-column h3 {
            font-size: 1.3rem;
            margin-bottom: 20px;
            position: relative;
            display: inline-block;
        }
        
        .footer-column h3::after {
            content: '';
            position: absolute;
            left: 0;
            bottom: -10px;
            width: 40px;
            height: 2px;
            background: var(--primary);
        }
        
        .footer-links {
            list-style: none;
        }
        
        .footer-links li {
            margin-bottom: 10px;
        }
        
        .footer-links a {
            color: #cbd5e1;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer-links a:hover {
            color: white;
        }
        
        .copyright {
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #334155;
            color: #94a3b8;
            font-size: 0.9rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .about-content {
                flex-direction: column;
            }
            
            .nav-links {
                display: none;
            }
            
            section {
                padding: 60px 0;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            h2 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header>
        <div class="container">
            <nav>
                <a href="#" class="logo">VaxInsight</a>
                <ul class="nav-links">
                    <li><a href="#about">About</a></li>
                    <li><a href="#features">Features</a></li>
                    <li><a href="#how-it-works">How It Works</a></li>
                    <li><a href="#cta" class="btn btn-outline">Get Started</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <!-- Hero Section -->
    <section id="hero">
        <div class="container">
            <h1>AI-Powered Vaccine Recommendations</h1>
            <p>Our advanced analytics platform identifies vaccine-hesitant populations and delivers personalized intervention strategies to improve vaccination rates and public health outcomes.</p>
            <a href="#cta" class="btn">Request Demo</a>
        </div>
    </section>

    <!-- About Section -->
    <section id="about">
        <div class="container">
            <h2>About VaxInsight</h2>
            <div class="about-content">
                <div class="about-text">
                    <p>VaxInsight is a cutting-edge public health analytics platform developed by a team of data scientists, epidemiologists, and behavioral psychologists. We combine artificial intelligence with behavioral science to combat vaccine hesitancy.</p>
                    <p>Our mission is to empower public health organizations with data-driven insights that enable targeted, effective vaccination campaigns. By understanding the root causes of vaccine hesitancy at both population and individual levels, we help create personalized interventions that work.</p>
                    <p>Since our founding in 2020, we've helped health departments and NGOs improve vaccination rates by an average of 27% in targeted populations.</p>
                </div>
                <div class="about-image">
                    <img src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1180&q=80" alt="Medical team discussing data">
                </div>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section id="features">
        <div class="container">
            <h2>Our Key Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Risk Group Identification</h3>
                    <p>Our algorithms pinpoint demographic groups with the highest vaccine hesitancy rates, allowing for targeted resource allocation.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h3>Behavioral Insights</h3>
                    <p>Understand the psychological and social factors driving hesitancy in different populations with our behavioral analysis engine.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3>Personalized Messaging</h3>
                    <p>Generate customized intervention messages tailored to address specific concerns and barriers to vaccination.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📈</div>
                    <h3>Impact Forecasting</h3>
                    <p>Predict the potential impact of different intervention strategies before implementation.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔄</div>
                    <h3>Continuous Learning</h3>
                    <p>Our system learns from each campaign, constantly improving its recommendations.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3>Privacy Protection</h3>
                    <p>All data is anonymized and processed with the highest security standards to protect individual privacy.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- How It Works -->
    <section id="how-it-works">
        <div class="container">
            <h2>How It Works</h2>
            <div class="steps">
                <div class="step">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <h3>Data Collection</h3>
                        <p>We integrate with your existing health data systems or surveys to gather vaccination and demographic information while maintaining strict privacy standards.</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <h3>Analysis & Insights</h3>
                        <p>Our AI models analyze the data to identify hesitancy patterns, risk factors, and behavioral drivers in your population.</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <h3>Recommendation Engine</h3>
                        <p>The system generates prioritized intervention recommendations tailored to each subgroup's specific barriers and concerns.</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">4</div>
                    <div class="step-content">
                        <h3>Implementation Support</h3>
                        <p>We provide tools and guidance to help you execute the recommended interventions effectively.</p>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">5</div>
                    <div class="step-content">
                        <h3>Impact Measurement</h3>
                        <p>Track the effectiveness of your campaigns and receive updated recommendations based on real-world results.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- CTA Section -->
    <section id="cta">
        <div class="container">
            <h2>Ready to Transform Your Vaccination Campaigns?</h2>
            <p>Join leading public health organizations using VaxInsight to make data-driven decisions that save lives.</p>
            <a href="\pages\1Preprocess.py" class="btn">Request a Demo</a>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="footer-content">
                <div class="footer-column">
                    <h3>VaxInsight</h3>
                    <p>AI-powered vaccine hesitancy solutions for public health organizations.</p>
                </div>
                <div class="footer-column">
                    <h3>Quick Links</h3>
                    <ul class="footer-links">
                        <li><a href="#about">About Us</a></li>
                        <li><a href="#features">Features</a></li>
                        <li><a href="#how-it-works">How It Works</a></li>
                        <li><a href="#cta">Get Started</a></li>
                    </ul>
                </div>
                <div class="footer-column">
                    <h3>Contact</h3>
                    <ul class="footer-links">
                        <li><a href="mailto:info@vaxinsight.com">info@vaxinsight.com</a></li>
                        <li><a href="tel:+18005551234">(800) 555-1234</a></li>
                        <li>123 Health Tech Way</li>
                        <li>San Francisco, CA 94107</li>
                    </ul>
                </div>
            </div>
            <div class="copyright">
                <p>&copy; 2025 VaxInsight. All rights reserved.</p>
            </div>
        </div>
    </footer>
</body>
</html>
"""


# Display the HTML in Streamlit
def main():
    st.set_page_config(layout="wide")
    html(html_content, width=None, height=2000, scrolling=True)

if __name__ == "__main__":
    main()