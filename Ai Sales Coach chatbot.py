import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Configure Gemini API Key
genai.configure(api_key="AIzaSyBj4-6YQVw0BBcVvk6frKavA5WVyKQ8Da8")
model = genai.GenerativeModel("gemini-1.5-pro")

def get_gemini_response(query):
    try:
        response = model.generate_content(query)
        return response.text.strip()
    except Exception as e:
        return f"Error getting response from Gemini 1.5: {e}"

def load_data():
    keywords = {
        'Cricket Equipment': [
            'Cricket Bat', 'MRF Bat', 'SG Bat', 'Types of Bats', 'Cricket Ball', 'Batting Gloves',
            'Cricket Helmet', 'Cricket Shoes', 'Bat Maintenance', 'Bat Size', 'Weight of Bat',
            'Cricket Kit'
        ],
        'Pricing and Offers': [
            'Price of MRF Bat', 'Price of SG Bat', 'Yonex Racket Price', 'Cricket Bat Price',
            'Product Warranty', 'Price Matching', 'Discount Offers', 'Clearance Sale',
            'Bulk Orders', 'Gift Cards'
        ],
        'Store Services': [
            'Store Location', 'Delivery Options', 'Custom Orders', 'Repair Services',
            'Equipment Rental', 'Personal Training', 'Gift Wrapping'
        ],
        'Sports Categories' : [
            'Badminton', 'Tennis', 'Football', 'Basketball', 'Running', 'Gym', 'Yoga',
            'Swimming', 'Camping', 'Cycling', 'Skating'
        ],
        'Equipment Types' : [
            'Tennis Racket', 'Football Boots', 'Basketball', 'Running Shoes', 'Helmet',
            'Sports Bag', 'Protective Gear', 'Sports Apparel', 'Sunglasses', 'Water Bottle',
            'Sports Watch', 'Fitness Tracker', 'Knee Pads', 'Sports Nutrition', 'Skates', 'Gym Shoes'
        ],
        'Special Features' : [
            'Product Authentication', 'Equipment Customization', 'Video Analysis',
            'Performance Tracking', 'Weather Resistance', 'Ergonomic Design'
        ]
    }

    answers = {
        'Cricket Equipment': [
            'We offer professional-grade bats from brands like MRF, SG, and Kookaburra, available in sizes 1-6. You can also try our swing analyzer to find the perfect bat.',
            'MRF bats are priced between ₹12,000 to ₹25,000, including a 1-year warranty. EMI options are available.',
            'SG bats range from ₹4,500 for training to ₹22,000 for international match quality bats.',
            'We provide various willow types: English for professionals, Kashmir for mid-level players, and Composite for beginners. Need personalized recommendations?',
            'We stock match balls such as the SG Test priced at ₹2,500 and Kookaburra Turf at ₹1,900. Practice balls start from ₹699.',
            'Our GM batting gloves feature impact foam and are priced at ₹1,200. We also offer custom sizing to ensure the perfect fit.',
            'We carry Masuri helmets starting from ₹6,500, equipped with a titanium grille. Visit us for an impact test demonstration.',
            'Our cricket shoes include SG spiked models with a 6-stud design, priced at ₹3,900. Replaceable spikes are available at ₹299 per set.',
            'Enjoy our free bat oiling service and maintain your bat with our willow conditioner, available at ₹599 per bottle.',
            'We offer bat sizes for adults (Short: 5f 6inch, Long: 5f 10inch ) and juniors. Visit us to find your ideal size.',
            'Standard bat weights range from 2.7 to 3.2 lbs, with MRF bats between 2.8 to 3.1 lbs, maintaining a ±0.05 lbs tolerance.',
            'Yes, we have full cricket kits that include bat, gloves, pads, helmet, and a kit bag — starting from ₹9,999.'
        ],
        'Pricing and Offers' : [
            'MRF bats are priced between ₹12,000 to ₹25,000, including a 1-year warranty. EMI options are available.',
            'SG bats range from ₹4,500 for training to ₹22,000 for international match quality bats.',
            'Yonex rackets are available from ₹1,500 to ₹7,000, with limited-time discounts of up to 15%.',
            'Cricket bats range from ₹2,500 for basic models to ₹35,000 for international-grade bats. Get fitted today!',
            'We offer a 1-year standard warranty on all products, extendable to 3 years for an additional ₹1,500, covering manufacturing defects.',
            'We match competitor prices by offering an additional 5% discount upon presenting valid proof.',
            'Our seasonal sale offers 20-40% discounts on previous models, with limited stock available.',
            'Enjoy up to 70% off on clearance items. Please note, these are final sale with no returns.',
            'Purchase 10 or more items to receive a 15% discount. Special rates are available for teams and coaches.',
            'Our eGift Cards are available in denominations from ₹1,000 to ₹50,000, with instant delivery and no expiration date.'
        ],
        'Store Services': [
            'Visit our flagship store at 123 Sports Street, featuring a batting cage and fitting stations.',
            'We offer free shipping on orders over ₹5,000. Express delivery within the city is available for ₹299, ensuring delivery within 2 hours.',
            'Customize your equipment with logos and colors. A 15-day lead time and a 50% deposit are required.',
            'Our bat restringing service is available at ₹799, with a 24-hour turnaround. Racquet restringing is ₹499.',
            'Rent camping gear at ₹999 per day, with a required security deposit.',
            'Book sessions with professional coaches at ₹2,500 per hour, including video analysis.',
            'Enjoy free gift wrapping services, with an option for a custom message card at ₹99.'
        ],
        'Sports Categories': [
            'We offer Yonex rackets and Li-Ning shuttles, with court booking services available.',
            'Choose from Wilson and Babolat rackets, with restringing services ranging from ₹300 to ₹800.',
            'Our selection includes Adidas and Nike football boots, with guidance on stud patterns suitable for your playing surface.',
            'Purchase official Spalding NBA basketballs starting at ₹4,500, available in both indoor and outdoor variants.',
            'Benefit from gait analysis when purchasing Nike or Asics running shoes, with complimentary insoles included.',
            'Explore our range of treadmills starting at ₹35,000, which includes free installation and a 5-year motor warranty.',
            'We provide eco-friendly yoga mats starting at ₹799, with free classes included with your purchase.',
            'Find Speedo swimming gear, including goggles starting at ₹999, with a free swim cap included.',
            'Our camping gear includes 4-season tents starting at ₹12,000, with a complimentary camping checklist provided.',
            'Explore road bikes starting at ₹25,000, which come with a free safety kit valued at ₹2,500.',
            'Rent roller skates starting at ₹3,500, with complimentary protective gear included.'
        ],
        'Equipment Types' : [
            'We offer tennis rackets from Yonex, Head, and Babolat, with demo sessions available in-store.',
            'Choose from Nike and Puma football boots designed for grass, turf, or indoor surfaces.',
            'NBA-approved basketballs start at ₹3,999, with combos including pump and net bag.',
            'Our running shoes include models from Adidas, Asics, and Reebok, starting at ₹2,999.',
            'We carry helmets for cycling, cricket, and skating, starting from ₹1,299 with ISI certification.',
            'Carry your gear in our ergonomic sports bags, starting at ₹999 with shoe compartments.',
            'Our protective gear includes elbow/knee guards and padded vests for contact sports.',
            'Get fitted for dry-fit sports apparel — shirts from ₹499, shorts from ₹599.',
            'Stay cool with our UV-blocking sunglasses for athletes, starting at ₹1,499.',
            'Stay hydrated with our BPA-free bottles, available in 1L and 2L sizes.',
            'All our sports watches come with GPS and heart rate tracking features.',
            'Our fitness trackers sync seamlessly with both iOS and Android.',
            'Knee pads with extra gel cushioning are available for high-impact sports.',
            'Sports nutrition packs include protein bars, electrolytes, and pre-workout drinks.',
            'Our skates come in adjustable sizes for kids and adults alike.',
            'Gym shoes from Adidas and Reebok start at ₹2,499 with free foot analysis to help you choose the best fit.'
        ],
        'Special Features': [
            'Each product includes a QR code for verifying authenticity through our app.',
            'We offer full customization on bats, shoes, and gloves. Choose from color, size, and logo options.',
            'Our advanced cameras provide swing and stance analysis with instant playback.',
            'Track performance with metrics like speed, impact, and consistency through our smart gear.',
            'Our gear is weather tested to ensure high performance in all conditions.',
            'Ergonomic design ensures better grip and reduced injury risk across all equipment types.'
        ]
    }

    data = []
    for category in keywords:
        for i, keyword in enumerate(keywords[category]):
            answer = answers.get(category, [])
            response = answer[i] if i < len(answer) else "Answer not available at the moment."
            data.append({'Keyword': keyword.lower(), 'Answer': response})

    return pd.DataFrame(data, columns=['Keyword', 'Answer'])

class SmartChatBot:
    def __init__(self):
        self.data = load_data()
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['Keyword'])
        self.context = {'current_topic': None}

    def is_sports_related(self, query):
        sports_terms = [
            'bat', 'ball', 'cricket', 'football', 'tennis', 'sports', 'gear', 'helmet',
            'fitness', 'shoes', 'running', 'basketball', 'badminton', 'racket', 'gym',
            'swimming', 'cycling', 'camping', 'yoga', 'skating',
            'volleyball', 'hockey', 'rugby', 'baseball', 'softball', 'track', 'field',
            'golf', 'surfing', 'skiing', 'snowboarding', 'boxing', 'wrestling', 'karate',
            'judo', 'taekwondo', 'martial arts', 'archery', 'fencing', 'rowing', 'climbing',
            'triathlon', 'weightlifting', 'bodybuilding', 'squash', 'polo', 'lacrosse',
            'ping pong', 'table tennis', 'diving', 'equestrian', 'cheerleading', 'parkour',
            'kitesurfing', 'wakeboarding', 'snorkeling', 'biking', 'trail running', 'hiking',
            'crossfit', 'pilates', 'aerobics', 'handball', 'freestyle', 'long jump',
            'high jump', 'pole vault', 'shot put', 'discus', 'kayaking', 'canoeing',
            'mountaineering', 'bouldering', 'slacklining', 'paragliding', 'motorsport',
            'dribbling', 'goalkeeping', 'kickboxing', 'stretching', 'warm-up', 'cool-down',
            'resistance bands', 'dumbbells', 'kettlebell', 'treadmill', 'elliptical',
            'jump rope', 'sportswear', 'jersey', 'shin guards', 'cleats', 'mouthguard',
            'goggles', 'wetsuit', 'hydration pack', 'stopwatch', 'referee', 'coach',
            'team', 'tournament', 'match', 'scoreboard', 'league', 'medal', 'trophy'
]

        return any(term in query.lower() for term in sports_terms)

    def process_query(self, query):
        query = query.strip().lower()

        greetings = ['hello', 'hi', 'hey', 'yo', 'sup', 'good morning', 'good afternoon', 'good evening']
        if any(greet == query for greet in greetings):
            return "Hey there! 👋 I'm your SportsPro Assistant. Ask me anything about sports gear, pricing, or services!"

        if self.context['current_topic'] and any(x in query for x in ['this', 'that', 'it']):
            query = f"{self.context['current_topic']} {query}"

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        best_match_idx = similarities.argmax()
        max_score = similarities.max()

        if max_score >= 0.6:
            result = self.data.iloc[best_match_idx]
            self.context['current_topic'] = result['Keyword']
            return result['Answer']
        elif self.is_sports_related(query):
            self.context['current_topic'] = None
            return get_gemini_response(query)
        else:
            return "Sorry, I can only help with sports and sports shop related questions."

def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .main-title {
            color: #4facfe;
            font-size: 2.8em !important;
            font-weight: 700 !important;
            text-align: center;
            padding: 5px 0;
            margin: 0 !important;
            text-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
            letter-spacing: 1px;
            position: relative;
            top: 20px;
        }
        .chat-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 20px;
            width: 800px;
            height: 70vh;
            overflow-y: auto;
            margin-top: 30px;
        }
        .sidebar-content {
            background: rgba(17, 34, 64, 0.95);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(79, 172, 254, 0.2);
            margin-bottom: 15px;
        }
        .equipment-card {
            background: rgba(26, 54, 93, 0.8);
            color: white;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
            border: 1px solid rgba(79, 172, 254, 0.2);
            cursor: pointer;
        }
        .equipment-card:hover {
            transform: translateY(-3px);
            background: rgba(79, 172, 254, 0.2);
            border-color: rgba(79, 172, 254, 0.4);
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.15);
        }
        .equipment-icon {
            font-size: 2em;
            text-align: center;
            margin-bottom: 8px;
            color: #4facfe;
        }
        .equipment-name {
            color: white;
            font-size: 1.2em;
            font-weight: 600;
            text-align: center;
            margin: 5px 0;
        }
        .equipment-action {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
            text-align: center;
            margin-top: 5px;
        }
        .stTextInput > div > div > input {
            background-color: rgba(26, 54, 93, 0.8);
            color: white;
            border-radius: 10px;
            border: 1px solid rgba(79, 172, 254, 0.3);
            padding: 12px 15px;
            font-size: 1em;
            margin-top: 10px;
        }
        .stTextInput > div > div > input:focus {
            border-color: rgba(79, 172, 254, 0.6);
            box-shadow: 0 0 15px rgba(79, 172, 254, 0.2);
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        .user-message {
            background-color: rgba(79, 172, 254, 0.1) !important;
            border-radius: 15px 15px 5px 15px !important;
            border: 1px solid rgba(79, 172, 254, 0.2);
            color: white !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            max-width: 85%;
            margin-left: auto !important;
        }
        .assistant-message {
            background-color: rgba(26, 54, 93, 0.6) !important;
            border-radius: 15px 15px 15px 5px !important;
            border: 1px solid rgba(79, 172, 254, 0.2);
            color: white !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            max-width: 85%;
        }
        .stMarkdown {
            color: white;
        }
        div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
            background-color: transparent !important;
        }
        .stChatMessage {
            background-color: transparent !important;
        }
        div[data-testid="stChatMessageContent"] {
            background-color: transparent !important;
            border: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Sports Sales Coach",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_custom_style()
    
    # Sidebar with sports equipment icons
    with st.sidebar:
        st.markdown('<h2 style="color: #4facfe; text-align: center; font-size: 1.8em; margin-bottom: 30px;">Our Sports Equipments</h2>', unsafe_allow_html=True)
        
        equipment_data = [
            {"icon": "🏏", "name": "Cricket", "action": "Playing cricket like a pro!"},
            {"icon": "🎾", "name": "Tennis", "action": "Serving aces!"},
            {"icon": "⚽", "name": "Football", "action": "Scoring goals!"},
            {"icon": "🏀", "name": "Basketball", "action": "Shooting hoops!"},
            {"icon": "🏸", "name": "Badminton", "action": "Smashing shuttles!"},
            {"icon": "⛳", "name": "Golf", "action": "Perfect swing!"}
        ]
        
        for item in equipment_data:
            st.markdown(f"""
                <div class="equipment-card">
                    <div class="equipment-icon">{item['icon']}</div>
                    <div class="equipment-name">{item['name']}</div>
                    <div class="equipment-action">{item['action']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Main chat interface
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Main title
        st.markdown('<h1 class="main-title">AI Sports Sales Coach 🏀</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{
                "role": "assistant",
                "content": "👋 Welcome to Sports Sales Coach! I'm your personal sports equipment expert. How can I assist you today?"
            }]
        
        # Display chat messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        bot = SmartChatBot()
        
        if prompt := st.chat_input("Ask me about sports equipment...", key="chat_input"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            
            with st.chat_message("assistant", avatar="🤖"):
                response = bot.process_query(prompt)
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
                st.rerun()

if __name__ == "__main__":
    main()
