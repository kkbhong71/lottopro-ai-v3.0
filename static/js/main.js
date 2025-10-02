/**
 * LottoPro-AI v3.0 - 메인 JavaScript 모듈
 * 개선된 버전 - UI 중복 문제 해결 및 모바일 메뉴 개선
 */

class LottoProApp {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.userPredictions = [];
        this.isLoading = false;
        this.algorithms = {};
        this.mobileMenuOpen = false;  // ✅ 추가
        
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
    
    // ===== 이벤트 리스너 설정 =====
    setupEventListeners() {
        // 테마 토글
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
        
        // 모바일 메뉴 - ✅ 개선된 버전
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenuBtn && mobileMenu) {
            mobileMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();  // 이벤트 버블링 방지
                this.toggleMobileMenu();
            });
            
            // 모바일 메뉴 링크 클릭 시 메뉴 닫기
            const menuLinks = mobileMenu.querySelectorAll('a');
            menuLinks.forEach(link => {
                link.addEventListener('click', () => {
                    this.closeMobileMenu();
                });
            });
        }
        
        // 플로팅 버튼
        const floatingBtn = document.getElementById('floating-btn');
        if (floatingBtn) {
            floatingBtn.addEventListener('click', () => this.handleFloatingAction());
        }
        
        // 페이지 가시성 변경 감지
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.refreshData();
            }
        });
        
        // 키보드 단축키
        document.addEventListener('keydown', (e) => {
            // ESC 키로 모바일 메뉴 닫기
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
    
    // ===== 테마 시스템 =====
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
        
        // 부드러운 전환 애니메이션
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
    
    // ===== 모바일 메뉴 - ✅ 완전히 개선된 버전 =====
    setupMobileMenu() {
        // 메뉴 외부 클릭 시 닫기
        document.addEventListener('click', (e) => {
            const mobileMenu = document.getElementById('mobile-menu');
            const mobileMenuBtn = document.getElementById('mobile-menu-btn');
            
            // 모바일 메뉴가 열려있고, 클릭한 곳이 메뉴나 버튼이 아니라면 닫기
            if (this.mobileMenuOpen && 
                mobileMenu && 
                !mobileMenu.contains(e.target) && 
                mobileMenuBtn && 
                !mobileMenuBtn.contains(e.target)) {
                this.closeMobileMenu();
            }
        });
        
        // 화면 크기 변경 시 데스크톱 모드면 메뉴 닫기
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
            mobileMenu.classList.remove('hidden');
            this.mobileMenuOpen = true;
            
            // 부드러운 애니메이션을 위해 약간의 지연
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
            
            // 애니메이션 후 hidden 클래스 추가
            setTimeout(() => {
                mobileMenu.classList.add('hidden');
                this.mobileMenuOpen = false;
            }, 300);
        }
    }
    
    // ✅ 기존 animateMenuToggle 함수 삭제 (더 이상 필요 없음)
    
    // ===== 토스트 알림 시스템 =====
    setupToastSystem() {
        // 토스트 컨테이너가 없으면 생성
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
        
        // 애니메이션으로 표시
        setTimeout(() => toast.classList.add('show'), 100);
        
        // 자동 제거
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
    
    // ===== 로딩 관리 =====
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
    
    // ===== 저장된 번호 관리 (개선) =====
    saveNumbers(numbers, algorithmName = 'AI 예측') {
        console.log('💾 번호 저장 시도:', numbers, algorithmName);
        
        try {
            // 번호 유효성 검사
            const validation = this.validateLottoNumbers(numbers);
            if (!validation.valid) {
                this.showToast(validation.message, 'error');
                return false;
            }
            
            // 기존 저장된 번호 로드
            const savedNumbers = this.getSavedNumbers();
            
            // 새 항목 생성
            const newEntry = {
                id: Date.now(),
                numbers: Array.isArray(numbers) ? numbers : [],
                timestamp: new Date().toISOString(),
                algorithm: algorithmName,
                checked: false,
                matches: 0
            };
            
            // 맨 앞에 추가
            savedNumbers.unshift(newEntry);
            
            // 저장 (최대 100개까지만 유지)
            const trimmedNumbers = savedNumbers.slice(0, 100);
            localStorage.setItem('savedNumbers', JSON.stringify(trimmedNumbers));
            
            console.log('✅ 저장 완료. 총 개수:', trimmedNumbers.length);
            
            // 성공 메시지
            this.showToast('번호가 저장되었습니다!', 'success');
            
            return true;
            
        } catch (error) {
            console.error('❌ 저장 실패:', error);
            this.showToast('번호 저장에 실패했습니다', 'error');
            return false;
        }
    }
    
    getSavedNumbers() {
        try {
            const data = localStorage.getItem('savedNumbers');
            return data ? JSON.parse(data) : [];
        } catch (error) {
            console.error('저장된 번호 로드 실패:', error);
            return [];
        }
    }
    
    deleteSavedNumber(id) {
        try {
            const savedNumbers = this.getSavedNumbers();
            const filtered = savedNumbers.filter(n => n.id !== id);
            localStorage.setItem('savedNumbers', JSON.stringify(filtered));
            this.showToast('번호가 삭제되었습니다', 'success');
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
    
    // ===== 데이터 관리 =====
    async loadUserData() {
        try {
            const response = await fetch('/api/user-predictions');
            const data = await response.json();
            this.userPredictions = data.predictions || [];
            this.updateStatsDisplay(data.stats || {});
        } catch (error) {
            console.error('사용자 데이터 로드 실패:', error);
        }
    }
    
    updateStatsDisplay(stats) {
        const elements = {
            'saved-predictions': stats.total_predictions || 0,
            'accuracy-rate': this.calculateAccuracy(stats) + '%',
            'best-match': stats.best_match || 0
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
    
    // ===== 실시간 업데이트 =====
    startRealTimeUpdates() {
        // 30초마다 데이터 새로고침
        setInterval(() => {
            this.updateLastUpdateTime();
        }, 30000);
        
        // 페이지 로드 시 시간 업데이트
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
    
    // ===== API 통신 =====
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
    
    // ===== 플로팅 액션 버튼 =====
    handleFloatingAction() {
        // 현재 페이지에 따라 다른 액션 수행
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
        // 첫 번째 알고리즘 실행 버튼 클릭
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
            
            // 시각적 강조 효과
            section.style.transition = 'all 0.5s ease';
            section.style.transform = 'scale(1.02)';
            section.style.boxShadow = '0 0 30px rgba(102, 126, 234, 0.3)';
            
            setTimeout(() => {
                section.style.transform = '';
                section.style.boxShadow = '';
            }, 1000);
        }
    }
    
    // ===== 유틸리티 함수 =====
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
    
    // ===== 데이터 새로고침 =====
    async refreshData() {
        if (this.isLoading) return;
        
        try {
            await this.loadUserData();
            this.showToast('데이터가 새로고침되었습니다', 'success', 2000);
        } catch (error) {
            console.error('데이터 새로고침 실패:', error);
        }
    }
    
    // ===== 검색 포커스 =====
    focusSearch() {
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="검색"]');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    // ===== 에러 처리 =====
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
    
    // ===== 성능 모니터링 =====
    logPerformance(operation, startTime) {
        const duration = performance.now() - startTime;
        console.log(`⚡ ${operation} 완료: ${duration.toFixed(2)}ms`);
        
        // 3초 이상 걸리는 작업은 경고
        if (duration > 3000) {
            console.warn(`⚠️ 느린 작업 감지: ${operation} (${duration.toFixed(2)}ms)`);
        }
    }
}

// ===== 전역 변수 및 초기화 =====
let lottoApp;

document.addEventListener('DOMContentLoaded', function() {
    // 앱 초기화
    lottoApp = new LottoProApp();
    
    // 전역 함수로 노출 (템플릿에서 사용)
    window.lottoApp = lottoApp;
    window.showToast = (msg, type, duration) => lottoApp.showToast(msg, type, duration);
    window.showLoading = (msg) => lottoApp.showLoading(msg);
    window.hideLoading = () => lottoApp.hideLoading();
    
    // 저장 관련 함수 (algorithm.html에서 사용)
    window.saveNumbers = (numbers, algorithmName) => lottoApp.saveNumbers(numbers, algorithmName);
    window.getSavedNumbers = () => lottoApp.getSavedNumbers();
    window.deleteSavedNumber = (id) => lottoApp.deleteSavedNumber(id);
    
    // PWA 설치 프롬프트
    setupPWAInstallPrompt();
    
    console.log('🎯 LottoPro-AI v3.0 준비 완료!');
});

// ===== PWA 설치 프롬프트 =====
function setupPWAInstallPrompt() {
    let deferredPrompt;
    
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        
        // 설치 버튼 표시 (필요시)
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
