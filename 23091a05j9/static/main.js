document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            const btn = form.querySelector('button[type="submit"]');
            btn.innerHTML = '<span class="pulse">Analyzing...</span>';
            btn.style.opacity = '0.8';
            btn.style.cursor = 'wait';
        });
    }
});
