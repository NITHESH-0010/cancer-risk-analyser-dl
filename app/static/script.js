// =========================================================
// ONCOVISION ML — PROFESSIONAL ANIMATIONS
// =========================================================


// =========================================================
// NAVBAR SCROLL EFFECT
// =========================================================

window.addEventListener("scroll", () => {

    const navbar = document.querySelector(".navbar");

    if (window.scrollY > 60) {

        navbar.style.background = "rgba(2,6,23,0.88)";
        navbar.style.backdropFilter = "blur(24px)";
        navbar.style.boxShadow = "0 10px 35px rgba(0,0,0,0.35)";

    } else {

        navbar.style.background = "rgba(2,6,23,0.45)";
        navbar.style.boxShadow = "none";

    }

});


// =========================================================
// SMOOTH SCROLL
// =========================================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {

    anchor.addEventListener("click", function (e) {

        e.preventDefault();

        const target = document.querySelector(this.getAttribute("href"));

        if (target) {

            target.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }

    });

});


// =========================================================
// FADE-IN REVEAL ANIMATION
// =========================================================

const revealElements = document.querySelectorAll(
    ".stat-card, .glass-card, .input-box, .info-card, .timeline-card, .prevention-card, .step-card, .result-container"
);

const revealObserver = new IntersectionObserver(

    (entries) => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {

                entry.target.classList.add("show-element");

            }

        });

    },

    {
        threshold: 0.15
    }

);

revealElements.forEach(el => {

    el.classList.add("hidden-element");

    revealObserver.observe(el);

});


// =========================================================
// INPUT GLOW EFFECT
// =========================================================

const inputs = document.querySelectorAll("input, select");

inputs.forEach(input => {

    input.addEventListener("focus", () => {

        input.parentElement.style.transform = "translateY(-5px)";
        input.parentElement.style.boxShadow =
            "0 10px 35px rgba(56,189,248,0.18)";

    });

    input.addEventListener("blur", () => {

        input.parentElement.style.transform = "translateY(0px)";
        input.parentElement.style.boxShadow = "none";

    });

});


// =========================================================
// BUTTON RIPPLE EFFECT
// =========================================================

const buttons = document.querySelectorAll(".predict-btn, .primary-btn");

buttons.forEach(button => {

    button.addEventListener("mouseenter", () => {

        button.style.transform = "translateY(-4px) scale(1.01)";

    });

    button.addEventListener("mouseleave", () => {

        button.style.transform = "translateY(0px) scale(1)";

    });

});


// =========================================================
// LIVE BMI COLOR INDICATOR
// =========================================================

const bmiInput = document.querySelector('input[name="BMI"]');

if (bmiInput) {

    bmiInput.addEventListener("input", () => {

        const value = parseFloat(bmiInput.value);

        if (!value) return;

        if (value < 18.5) {

            bmiInput.style.border = "2px solid #facc15";

        }

        else if (value >= 18.5 && value <= 24.9) {

            bmiInput.style.border = "2px solid #22c55e";

        }

        else if (value >= 25 && value <= 29.9) {

            bmiInput.style.border = "2px solid #f97316";

        }

        else {

            bmiInput.style.border = "2px solid #ef4444";

        }

    });

}


// =========================================================
// PREDICT BUTTON LOADING EFFECT
// =========================================================

const form = document.querySelector(".prediction-form");

const predictBtn = document.querySelector(".predict-btn");

if (form && predictBtn) {

    form.addEventListener("submit", () => {

        predictBtn.innerHTML = "Analyzing Risk...";

        predictBtn.style.opacity = "0.85";

        predictBtn.style.pointerEvents = "none";

    });

}


// =========================================================
// HERO FLOATING ANIMATION
// =========================================================

const heroCard = document.querySelector(".glass-card");

if (heroCard) {

    let floating = 0;

    setInterval(() => {

        floating += 0.02;

        heroCard.style.transform =
            `translateY(${Math.sin(floating) * 8}px)`;

    }, 20);

}


// =========================================================
// PARTICLE GLOW EFFECT
// =========================================================

function createParticle() {

    const particle = document.createElement("div");

    particle.classList.add("particle");

    document.body.appendChild(particle);

    const size = Math.random() * 5 + 2;

    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;

    particle.style.left = Math.random() * window.innerWidth + "px";

    particle.style.animationDuration =
        Math.random() * 10 + 8 + "s";

    particle.style.opacity = Math.random();

    setTimeout(() => {

        particle.remove();

    }, 12000);

}

setInterval(createParticle, 450);


// =========================================================
// TYPING EFFECT
// =========================================================

const heroTitle = document.querySelector(".hero-left h1");

if (heroTitle) {

    const originalHTML = heroTitle.innerHTML;

    const tempText = heroTitle.textContent;

    heroTitle.innerHTML = "";

    let index = 0;

    const typing = setInterval(() => {

        heroTitle.innerHTML =
            originalHTML.substring(0, index);

        index++;

        if (index > originalHTML.length) {

            clearInterval(typing);

        }

    }, 35);

}


// =========================================================
// ACTIVE NAVIGATION HIGHLIGHT
// =========================================================

const sections = document.querySelectorAll("section");

const navLinks = document.querySelectorAll(".nav-links a");

window.addEventListener("scroll", () => {

    let current = "";

    sections.forEach(section => {

        const sectionTop = section.offsetTop - 150;

        if (pageYOffset >= sectionTop) {

            current = section.getAttribute("id");

        }

    });

    navLinks.forEach(link => {

        link.classList.remove("active-nav");

        if (link.getAttribute("href") === `#${current}`) {

            link.classList.add("active-nav");

        }

    });

});


// =========================================================
// PARALLAX BACKGROUND EFFECT
// =========================================================

window.addEventListener("scroll", () => {

    const bg =
        document.querySelector(".background-wrapper img");

    const offset = window.pageYOffset;

    bg.style.transform =
        `translateY(${offset * 0.12}px) scale(1.05)`;

});


// =========================================================
// MOUSE GLOW EFFECT
// =========================================================

const glow = document.createElement("div");

glow.classList.add("mouse-glow");

document.body.appendChild(glow);

document.addEventListener("mousemove", (e) => {

    glow.style.left = e.clientX + "px";

    glow.style.top = e.clientY + "px";

});


// =========================================================
// SUCCESSFUL LOAD ANIMATION
// =========================================================

window.addEventListener("load", () => {

    document.body.classList.add("loaded");

});


// =========================================================
// CONSOLE MESSAGE
// =========================================================

console.log(
    "%cOncoVision ML Loaded Successfully",
    "color:#38bdf8; font-size:18px; font-weight:bold;"
);