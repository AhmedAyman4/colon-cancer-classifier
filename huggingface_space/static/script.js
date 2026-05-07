// Element References
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const resultsSection = document.getElementById('results-section');
const mainPrediction = document.getElementById('main-prediction');
const mainConfidence = document.getElementById('main-confidence');
const probBars = document.getElementById('prob-bars');
const loader = document.getElementById('loader');
const btnText = document.getElementById('btn-text');
const navLinks = document.querySelectorAll('.nav-links a');

let selectedFile = null;

// --- GSAP Animations ---

// Register ScrollTrigger
gsap.registerPlugin(ScrollTrigger);

// Hero Entrance
const heroTitle = document.querySelector('.hero-section .display-text');
const titleText = heroTitle.textContent;
heroTitle.innerHTML = titleText.split('').map(char => `<span class="char">${char === ' ' ? '&nbsp;' : char}</span>`).join('');

gsap.from('.char', {
    y: 100,
    opacity: 0,
    duration: 0.8,
    stagger: 0.05,
    ease: "back.out(1.7)",
    delay: 0.2
});

gsap.from('.hero-section .body-text', {
    y: 30,
    opacity: 0,
    duration: 1,
    ease: "power4.out",
    delay: 1
});

// Inline Nav Entrance
gsap.from('.inline-nav', {
    y: -20,
    opacity: 0,
    duration: 1,
    ease: "power4.out"
});

// Card Entrance
gsap.from('.main-card', {
    y: 60,
    opacity: 0,
    duration: 1.2,
    ease: "power4.out",
    delay: 0.6
});

// --- Interactivity ---

// Active Link Tracking (Simplified)
window.addEventListener('scroll', () => {
    let current = "";
    document.querySelectorAll('section, header, footer').forEach((section) => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 100) {
            current = section.getAttribute("id");
        }
    });

    navLinks.forEach((link) => {
        link.classList.remove("active");
        if (link.getAttribute("href").includes(current)) {
            link.classList.add("active");
        }
    });
});

// File Upload Logic
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        
        // Animate transition
        gsap.to(dropZone, {
            opacity: 0,
            scale: 0.95,
            duration: 0.4,
            onComplete: () => {
                dropZone.style.display = 'none';
                previewContainer.style.display = 'flex';
                resultsSection.style.display = 'none';
                gsap.fromTo(previewContainer, { opacity: 0, scale: 1.05 }, { opacity: 1, scale: 1, duration: 0.5 });
            }
        });
    };
    reader.readAsDataURL(file);
}

resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '#';
    
    gsap.to(previewContainer, {
        opacity: 0,
        scale: 0.95,
        duration: 0.4,
        onComplete: () => {
            previewContainer.style.display = 'none';
            dropZone.style.display = 'block';
            resultsSection.style.display = 'none';
            gsap.fromTo(dropZone, { opacity: 0, scale: 1.05 }, { opacity: 1, scale: 1, duration: 0.5 });
        }
    });
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loader.style.display = 'inline-block';
    btnText.textContent = 'Processing...';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error(error);
        alert('Error: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        loader.style.display = 'none';
        btnText.textContent = 'Analyze Image';
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';
    mainPrediction.textContent = data.display_name;
    mainConfidence.textContent = `${data.confidence}% Confidence`;

    mainPrediction.className = 'prediction-label display-text';
    if (data.prediction === 'colon_aca') {
        mainPrediction.classList.add('adenocarcinoma');
    } else {
        mainPrediction.classList.add('benign');
    }

    probBars.innerHTML = '';
    Object.entries(data.probabilities).forEach(([name, prob]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';
        barContainer.innerHTML = `
            <div class="bar-label">
                <span>${name}</span>
                <span>${prob}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 0%"></div>
            </div>
        `;
        probBars.appendChild(barContainer);

        // Bar animation handled by CSS transition
        setTimeout(() => {
            barContainer.querySelector('.bar-fill').style.width = `${prob}%`;
        }, 100);
    });

    // Animate results entry
    gsap.from(resultsSection, {
        y: 30,
        opacity: 0,
        duration: 0.8,
        ease: "power3.out"
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

