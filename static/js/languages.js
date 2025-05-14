const translations = {
    en: {
        // Navigation
        about: "About",
        contact: "Contact",
        
        // Sidebar
        faq: "FAQ",
        noFaqsFound: "No FAQs found",
        
        // Chat
        typeMessage: "Type your message here...",
        thinking: "Thinking...",
        helpful: "Helpful",
        notHelpful: "Not Helpful",
        feedbackRecorded: "Feedback recorded",
        thankYouFeedback: "Thank you for your feedback!",
        
        // Error Messages
        errorOccurred: "An error occurred while processing your question",
        errorFeedback: "An error occurred while submitting feedback",
        failedFeedback: "Failed to record feedback",
        sorryError: "Sorry, an error occurred. Please try again.",
        
        // Sources
        source: "Source:",
        
        // Language Names
        english: "English",
        finnish: "Finnish"
    },
    fi: {
        // Navigation
        about: "Tietoa",
        contact: "Yhteystiedot",
        
        // Sidebar
        faq: "UKK",
        noFaqsFound: "UKK:ita ei löytynyt",
        
        // Chat
        typeMessage: "Kirjoita viestisi tähän...",
        thinking: "Ajattelen...",
        helpful: "Hyödyllinen",
        notHelpful: "Ei hyödyllinen",
        feedbackRecorded: "Palaute tallennettu",
        thankYouFeedback: "Kiitos palautteestasi!",
        
        // Error Messages
        errorOccurred: "Kysymyksen käsittelyssä tapahtui virhe",
        errorFeedback: "Palautteen lähetyksessä tapahtui virhe",
        failedFeedback: "Palautteen tallennus epäonnistui",
        sorryError: "Valitettavasti tapahtui virhe. Yritä uudelleen.",
        
        // Sources
        source: "Lähde:",
        
        // Language Names
        english: "Englanti",
        finnish: "Suomi"
    }
};

// Function to get translation
function getTranslation(key, lang = 'en') {
    return translations[lang]?.[key] || translations['en'][key] || key;
}

// Function to update all translations on the page
function updateTranslations(lang) {
    // Update all elements with data-translate attribute
    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.getAttribute('data-translate');
        element.textContent = getTranslation(key, lang);
    });
    
    // Update placeholders
    document.querySelectorAll('[data-translate-placeholder]').forEach(element => {
        const key = element.getAttribute('data-translate-placeholder');
        element.placeholder = getTranslation(key, lang);
    });
    
    // Store selected language in localStorage
    localStorage.setItem('selectedLanguage', lang);
}

// Function to initialize language
function initializeLanguage() {
    const savedLanguage = localStorage.getItem('selectedLanguage') || 'en';
    updateTranslations(savedLanguage);
    return savedLanguage;
} 