/**
 * LottoPro-AI v3.0 - 메인 JavaScript 모듈
 * 개선된 버전 - 백엔드 API 통합
 */

class LottoProApp {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.userPredictions = [];
        this.isLoading = false;
        this.algorithms = {};
        this.mobileMenuOpen = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupTheme();
        this.setupMobileMenu();
        this.setupToastSystem();
        this.loadUserData();
        this.startRealTimeUpdates();
        
        console.log('🚀 LottoPro-AI v3.0 초기화 완료');
    }
    
    setupEventListeners() {
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
        
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenuBtn && mobileMenu) {
            mobileMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleMobileMenu();
            });
            
            const menuLinks = mobileMenu.querySelectorAll('a');
            menuLinks.forEach(link => {
                link.addEventListener('click', () => {
                    this.closeMobileMenu();
                });
            });
        }
        
        const floatingBtn = document.getElementById('floating-btn');
        if (floatingBtn) {
            floatingBtn.addEventListener('click', () => this.handleFloatingAction());
        }
        
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.refreshData();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.mobileMenuOpen) {
                this.closeMobileMenu();
            }
            
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'd':
                        e.preventDefault();
                        this.toggleTheme();
                        break;
                    case 'k':
                        e.preventDefault();
                        this.focusSearch();
                        break;
                }
            }
        });
    }
    
    setupTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        this.updateThemeIcon();
    }
    
    toggleTheme() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
        this.updateThemeIcon();
        this.showToast('테마가 변경되었습니다', 'success');
        
        document.body.style.transition = 'all 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }
    
    updateThemeIcon() {
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) {
            themeIcon.className = this.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
    
    setupMobileMenu() {
        document.addEventListener('click', (e) => {
            const mobileMenu = document.getElementById('mobile-menu');
            const mobileMenuBtn = document.getElementById('mobile-menu-btn');
            
            if (this.mobileMenuOpen && 
                mobileMenu && 
                !mobileMenu.contains(e.target) && 
                mobileMenuBtn && 
                !mobileMenuBtn.contains(e.target)) {
                this.closeMobileMenu();
            }
        });
        
        window.addEventListener('resize', () => {
            if (window.innerWidth >= 768 && this.mobileMenuOpen) {
                this.closeMobileMenu();
            }
        });
    }
    
    toggleMobileMenu() {
        if (this.mobileMenuOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }
    
    openMobileMenu() {
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenu) {
            mobileMenu.style.display = 'block';
            mobileMenu.classList.remove('hidden');
            this.mobileMenuOpen = true;
            
            setTimeout(() => {
                mobileMenu.style.opacity = '1';
                mobileMenu.style.transform = 'translateY(0)';
            }, 10);
        }
    }
    
    closeMobileMenu() {
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenu) {
            mobileMenu.style.opacity = '0';
            mobileMenu.style.transform = 'translateY(-10px)';
            
            setTimeout(() => {
                mobileMenu.classList.add('hidden');
                mobileMenu.style.display = 'none';
                this.mobileMenuOpen = false;
            }, 300);
        }
    }
    
    setupToastSystem() {
        if (!document.getElementById('toast-container')) {
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'fixed top-20 right-6 z-50 space-y-3';
            document.body.appendChild(container);
        }
    }
    
    showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `
            <div class="flex items-center">
                <i class="${icon} mr-3"></i>
                <span>${message}</span>
                <button class="ml-4 opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
    
    getToastIcon(type) {
        const icons = {
            success: 'fas fa-check-circle text-green-400',
            error: 'fas fa-exclamation-circle text-red-400',
            warning: 'fas fa-exclamation-triangle text-yellow-400',
            info: 'fas fa-info-circle text-blue-400'
        };
        return icons[type] || icons.info;
    }
    
    showLoading(message = 'AI가 번호를 분석중입니다...') {
        this.isLoading = true;
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        
        if (loadingText) loadingText.textContent = message;
        if (overlay) {
            overlay.classList.remove('hidden');
            overlay.style.opacity = '0';
            setTimeout(() => overlay.style.opacity = '1', 10);
        }
    }
    
    hideLoading() {
        this.isLoading = false;
        const overlay = document.getElementById('loading-overlay');
        
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.classList.add('hidden'), 300);
        }
    }
    
    async saveNumbers(numbers, algorithmName = 'AI 예측', algorithmId = 'unknown') {
        console.log('💾 번호 저장 시도:', numbers, algorithmName, algorithmId);
        
        try {
            const validation = this.validateLottoNumbers(numbers);
            if (!validation.valid) {
                this.showToast(validation.message, 'error');
                return false;
            }
            
            // 백엔드 API로 저장
            const response = await fetch('/api/save-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    numbers: Array.isArray(numbers) ? numbers : [],
                    algorithm: algorithmId,
                    algorithm_name: algorithmName,
                    timestamp: new Date().toISOString(),
                    round_predicted: 1191
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // localStorage에도 캐시 (선택적)
                this.cacheToLocalStorage(numbers, algorithmName);
                
                console.log('✅ 서버 저장 완료:', result.prediction_id);
                this.showToast('번호가 저장되었습니다!', 'success');
                
                // 데이터 새로고침
                await this.loadUserData();
                
                return true;
            } else {
                throw new Error(result.message || '저장 실패');
            }
            
        } catch (error) {
            console.error('❌ 저장 실패:', error);
            this.showToast('번호 저장에 실패했습니다', 'error');
            return false;
        }
    }
    
    cacheToLocalStorage(numbers, algorithmName) {
        try {
            const cached = JSON.parse(localStorage.getItem('savedNumbers') || '[]');
            cached.unshift({
                id: Date.now(),
                numbers: numbers,
                timestamp: new Date().toISOString(),
                algorithm: algorithmName,
                cached: true
            });
            localStorage.setItem('savedNumbers', JSON.stringify(cached.slice(0, 50)));
        } catch (error) {
            console.warn('localStorage 캐시 실패:', error);
        }
    }
    
    getSavedNumbers() {
        // localStorage가 아닌 서버에서 가져오기
        return this.userPredictions || [];
    }
    
    async deleteSavedNumber(id) {
        try {
            // 서버의 prediction_id로 삭제하는 API가 필요함
            // 현재는 localStorage에서만 삭제
            const savedNumbers = JSON.parse(localStorage.getItem('savedNumbers') || '[]');
            const filtered = savedNumbers.filter(n => n.id !== id);
            localStorage.setItem('savedNumbers', JSON.stringify(filtered));
            
            this.showToast('번호가 삭제되었습니다', 'success');
            await this.loadUserData();
            return true;
        } catch (error) {
            console.error('삭제 실패:', error);
            this.showToast('삭제에 실패했습니다', 'error');
            return false;
        }
    }
    
    clearAllSavedNumbers() {
        try {
            localStorage.removeItem('savedNumbers');
            this.showToast('모든 번호가 삭제되었습니다', 'success');
            return true;
        } catch (error) {
            console.error('전체 삭제 실패:', error);
            return false;
        }
    }
    
    async loadUserData() {
        try {
            const response = await fetch('/api/user-predictions');
            const data = await response.json();
            
            this.userPredictions = data.predictions || [];
            this.updateStatsDisplay(data.stats || {});
            
            console.log('📊 사용자 데이터 로드:', this.userPredictions.length, '개');
            
            return data;
        } catch (error) {
            console.error('사용자 데이터 로드 실패:', error);
            return { predictions: [], stats: {} };
        }
    }
    
    updateStatsDisplay(stats) {
        const elements = {
            'saved-predictions': stats.total_predictions || 0,
            'total-predictions': stats.total_predictions || 0,
            'accuracy-rate': this.calculateAccuracy(stats) + '%',
            'best-match': stats.best_match || 0,
            'total-matches': stats.total_matches || 0
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                this.animateNumber(element, value);
            }
        });
    }
    
    calculateAccuracy(stats) {
        if (!stats.total_predictions || stats.total_predictions === 0) return 0;
        return Math.round((stats.total_matches / (stats.total_predictions * 6)) * 100);
    }
    
    animateNumber(element, targetValue) {
        const currentValue = parseInt(element.textContent) || 0;
        const target = parseInt(targetValue) || 0;
        const duration = 1000;
        const steps = 30;
        const increment = (target - currentValue) / steps;
        
        let current = currentValue;
        let step = 0;
        
        const timer = setInterval(() => {
            step++;
            current += increment;
            element.textContent = Math.round(current);
            
            if (step >= steps) {
                element.textContent = targetValue;
                clearInterval(timer);
            }
        }, duration / steps);
    }
    
    startRealTimeUpdates() {
        setInterval(() => {
            this.updateLastUpdateTime();
        }, 30000);
        
        this.updateLastUpdateTime();
    }
    
    updateLastUpdateTime() {
        const element = document.getElementById('last-update');
        if (element) {
            const now = new Date();
            const timeString = this.getRelativeTime(now);
            element.textContent = timeString;
        }
    }
    
    getRelativeTime(date) {
        const now = new Date();
        const diffInSeconds = Math.floor((now - date) / 1000);
        
        if (diffInSeconds < 60) return '방금';
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}분 전`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}시간 전`;
        return `${Math.floor(diffInSeconds / 86400)}일 전`;
    }
    
    async apiRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API 요청 실패:', error);
            this.showToast('서버 통신 중 오류가 발생했습니다', 'error');
            throw error;
        }
    }
    
    handleFloatingAction() {
        const path = window.location.pathname;
        
        switch (path) {
            case '/':
                this.scrollToSection('quick-prediction-section');
                break;
            case '/algorithms':
                this.quickAlgorithmRun();
                break;
            case '/saved-numbers':
                this.addNewPrediction();
                break;
            case '/compare':
                this.quickCompare();
                break;
            default:
                window.location.href = '/algorithms';
        }
    }
    
    quickAlgorithmRun() {
        const firstBtn = document.querySelector('[data-algorithm-id]');
        if (firstBtn) {
            firstBtn.click();
        } else {
            this.showToast('실행 가능한 알고리즘이 없습니다', 'warning');
        }
    }
    
    addNewPrediction() {
        window.location.href = '/algorithms';
    }
    
    quickCompare() {
        this.showToast('당첨번호를 입력해주세요', 'info');
    }
    
    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
            
            section.style.transition = 'all 0.5s ease';
            section.style.transform = 'scale(1.02)';
            section.style.boxShadow = '0 0 30px rgba(102, 126, 234, 0.3)';
            
            setTimeout(() => {
                section.style.transform = '';
                section.style.boxShadow = '';
            }, 1000);
        }
    }
    
    generateRandomNumbers(count = 6, min = 1, max = 45) {
        const numbers = [];
        while (numbers.length < count) {
            const num = Math.floor(Math.random() * (max - min + 1)) + min;
            if (!numbers.includes(num)) {
                numbers.push(num);
            }
        }
        return numbers.sort((a, b) => a - b);
    }
    
    formatNumbers(numbers) {
        return numbers.map(num => num.toString().padStart(2, '0')).join(', ');
    }
    
    validateLottoNumbers(numbers) {
        if (!Array.isArray(numbers)) {
            return { valid: false, message: '번호는 배열이어야 합니다' };
        }
        
        if (numbers.length !== 6) {
            return { valid: false, message: '6개의 번호가 필요합니다' };
        }
        
        for (const num of numbers) {
            if (!Number.isInteger(num) || num < 1 || num > 45) {
                return { valid: false, message: '번호는 1-45 범위여야 합니다' };
            }
        }
        
        if (new Set(numbers).size !== numbers.length) {
            return { valid: false, message: '중복된 번호가 있습니다' };
        }
        
        return { valid: true };
    }
    
    async refreshData() {
        if (this.isLoading) return;
        
        try {
            await this.loadUserData();
            this.showToast('데이터가 새로고침되었습니다', 'success', 2000);
        } catch (error) {
            console.error('데이터 새로고침 실패:', error);
        }
    }
    
    focusSearch() {
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="검색"]');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    handleError(error, context = '') {
        console.error(`오류 발생 ${context}:`, error);
        
        let message = '알 수 없는 오류가 발생했습니다';
        
        if (error.message) {
            if (error.message.includes('fetch')) {
                message = '네트워크 연결을 확인해주세요';
            } else if (error.message.includes('404')) {
                message = '요청한 리소스를 찾을 수 없습니다';
            } else if (error.message.includes('500')) {
                message = '서버 오류가 발생했습니다';
            } else {
                message = error.message;
            }
        }
        
        this.showToast(message, 'error');
    }
    
    logPerformance(operation, startTime) {
        const duration = performance.now() - startTime;
        console.log(`⚡ ${operation} 완료: ${duration.toFixed(2)}ms`);
        
        if (duration > 3000) {
            console.warn(`⚠️ 느린 작업 감지: ${operation} (${duration.toFixed(2)}ms)`);
        }
    }
}

let lottoApp;

document.addEventListener('DOMContentLoaded', function() {
    lottoApp = new LottoProApp();
    
    window.lottoApp = lottoApp;
    window.showToast = (msg, type, duration) => lottoApp.showToast(msg, type, duration);
    window.showLoading = (msg) => lottoApp.showLoading(msg);
    window.hideLoading = () => lottoApp.hideLoading();
    
    window.saveNumbers = (numbers, algorithmName, algorithmId) => lottoApp.saveNumbers(numbers, algorithmName, algorithmId);
    window.getSavedNumbers = () => lottoApp.getSavedNumbers();
    window.deleteSavedNumber = (id) => lottoApp.deleteSavedNumber(id);
    
    setupPWAInstallPrompt();
    
    console.log('🎯 LottoPro-AI v3.0 준비 완료!');
});

function setupPWAInstallPrompt() {
    let deferredPrompt;
    
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        
        const installBtn = document.getElementById('pwa-install-btn');
        if (installBtn) {
            installBtn.style.display = 'block';
            installBtn.addEventListener('click', () => {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('PWA 설치됨');
                        lottoApp.showToast('앱이 설치되었습니다!', 'success');
                    }
                    deferredPrompt = null;
                });
            });
        }
    });
    
    window.addEventListener('appinstalled', () => {
        console.log('PWA 설치 완료');
        lottoApp.showToast('LottoPro-AI가 성공적으로 설치되었습니다!', 'success', 5000);
    });
}
