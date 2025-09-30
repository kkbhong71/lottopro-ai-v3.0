/**
 * GitHub API 연동 모듈 v3.0 - 이벤트 위임 방식
 * 알고리즘 코드 실행 및 관리
 */

class GitHubAPIManager {
    constructor() {
        this.baseURL = '/api';
        this.algorithms = {};
        this.executionHistory = [];
        this.isInitialized = false;
        this.initPromise = null;
        
        console.log('GitHubAPIManager 생성됨');
    }
    
    async init() {
        if (this.initPromise) {
            return this.initPromise;
        }
        
        this.initPromise = (async () => {
            try {
                console.log('알고리즘 정보 로드 시작...');
                await this.loadAlgorithmInfo();
                this.setupAlgorithmCards();
                this.isInitialized = true;
                console.log('GitHubAPIManager 초기화 완료');
            } catch (error) {
                console.error('GitHubAPIManager 초기화 실패:', error);
                this.isInitialized = false;
            }
        })();
        
        return this.initPromise;
    }
    
    async ensureInitialized() {
        if (!this.isInitialized) {
            await this.init();
        }
    }
    
    // ===== 알고리즘 정보 로드 =====
    async loadAlgorithmInfo() {
        try {
            console.log('API 호출: /api/algorithm-info');
            
            const response = await fetch('/api/algorithm-info');
            
            if (!response.ok) {
                throw new Error(`API 응답 실패: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API 응답 받음:', data);
            
            if (data.status === 'success' && data.info && data.info.algorithms) {
                this.algorithms = data.info.algorithms;
                console.log(`알고리즘 ${Object.keys(this.algorithms).length}개 로드 완료`);
                console.log('로드된 알고리즘:', Object.keys(this.algorithms));
            } else {
                console.warn('예상치 못한 API 응답 구조:', data);
                throw new Error('알고리즘 정보 구조가 올바르지 않습니다');
            }
            
        } catch (error) {
            console.error('알고리즘 정보 로드 실패:', error);
            console.warn('폴백 알고리즘 사용');
            this.algorithms = this.getFallbackAlgorithms();
        }
    }
    
    getFallbackAlgorithms() {
        return {
            fallback_1: {
                id: 'fallback_1',
                name: "기본 알고리즘 1",
                subtitle: "폴백 모드",
                description: "서버 연결 문제로 인한 기본 알고리즘",
                accuracy: "N/A",
                icon: "⚠️",
                color: "#FF6B6B",
                complexity: "Low",
                execution_time: "~1초",
                features: ["오프라인 모드", "기본 랜덤 생성"]
            }
        };
    }
    
    // ===== 알고리즘 카드 설정 =====
    setupAlgorithmCards() {
        const previewContainer = document.getElementById('algorithm-preview');
        const algorithmGrid = document.getElementById('algorithm-grid');
        
        if (previewContainer) {
            console.log('미리보기 카드 렌더링 시작');
            this.renderAlgorithmPreview(previewContainer);
        }
        
        if (algorithmGrid) {
            console.log('전체 카드 그리드 렌더링 시작');
            this.renderAlgorithmGrid(algorithmGrid);
        }
    }
    
    renderAlgorithmPreview(container) {
        container.innerHTML = '';
        
        const topAlgorithms = Object.entries(this.algorithms).slice(0, 4);
        
        if (topAlgorithms.length === 0) {
            console.warn('표시할 알고리즘이 없습니다');
            container.innerHTML = '<p class="text-white">알고리즘을 로드할 수 없습니다.</p>';
            return;
        }
        
        topAlgorithms.forEach(([id, algorithm]) => {
            const card = this.createAlgorithmPreviewCard(id, algorithm);
            container.appendChild(card);
        });
        
        this.animateCards(container.children);
        console.log(`미리보기 카드 ${topAlgorithms.length}개 렌더링 완료`);
        
        // 이벤트 위임 설정
        container.addEventListener('click', (e) => {
            const quickRunBtn = e.target.closest('.quick-run-btn');
            if (quickRunBtn) {
                e.preventDefault();
                e.stopPropagation();
                const algorithmId = quickRunBtn.dataset.algorithm;
                console.log(`빠른 실행 클릭: ${algorithmId}`);
                this.executeAlgorithm(algorithmId);
            }
        });
    }
    
    renderAlgorithmGrid(container) {
        container.innerHTML = '';
        
        const algorithms = Object.entries(this.algorithms);
        
        if (algorithms.length === 0) {
            console.warn('표시할 알고리즘이 없습니다');
            container.innerHTML = '<p class="text-white text-center">알고리즘을 로드할 수 없습니다.</p>';
            return;
        }
        
        algorithms.forEach(([id, algorithm]) => {
            const card = this.createAlgorithmDetailCard(id, algorithm);
            container.appendChild(card);
        });
        
        this.animateCards(container.children);
        console.log(`전체 카드 ${algorithms.length}개 렌더링 완료`);
        
        // 이벤트 위임으로 모든 버튼 클릭 처리
        container.addEventListener('click', (e) => {
            // 실행 버튼 클릭
            const executeBtn = e.target.closest('.btn-execute');
            if (executeBtn) {
                e.preventDefault();
                e.stopPropagation();
                const algorithmId = executeBtn.dataset.algorithm;
                const card = executeBtn.closest('.algorithm-detail-card');
                console.log(`실행 버튼 클릭 (위임): ${algorithmId}`);
                this.executeAlgorithm(algorithmId, card);
                return;
            }
            
            // 정보 버튼 클릭
            const infoBtn = e.target.closest('.btn-info');
            if (infoBtn) {
                e.preventDefault();
                e.stopPropagation();
                const algorithmId = infoBtn.dataset.algorithm;
                const algorithm = this.algorithms[algorithmId];
                console.log(`정보 버튼 클릭 (위임): ${algorithmId}`);
                this.showAlgorithmInfo(algorithmId, algorithm);
                return;
            }
            
            // 결과 닫기 버튼
            const closeBtn = e.target.closest('.result-close');
            if (closeBtn) {
                e.preventDefault();
                e.stopPropagation();
                const resultContainer = closeBtn.closest('.execution-result');
                if (resultContainer) {
                    const algorithmId = resultContainer.id.replace('result-', '');
                    console.log(`결과 닫기 클릭: ${algorithmId}`);
                    this.hideResult(algorithmId);
                }
                return;
            }
            
            // 저장 버튼
            const saveBtn = e.target.closest('.btn-save-result');
            if (saveBtn) {
                e.preventDefault();
                e.stopPropagation();
                const numbers = saveBtn.dataset.numbers.split(',').map(n => parseInt(n));
                const algorithmId = saveBtn.dataset.algorithm;
                console.log(`저장 버튼 클릭: ${algorithmId}`);
                this.savePrediction(numbers, algorithmId);
                return;
            }
            
            // 복사 버튼
            const copyBtn = e.target.closest('.btn-copy-result');
            if (copyBtn) {
                e.preventDefault();
                e.stopPropagation();
                const numbers = copyBtn.dataset.numbers;
                console.log(`복사 버튼 클릭`);
                this.copyToClipboard(numbers);
                return;
            }
            
            // 재시도 버튼
            const retryBtn = e.target.closest('.btn-retry');
            if (retryBtn) {
                e.preventDefault();
                e.stopPropagation();
                const algorithmId = retryBtn.dataset.algorithm;
                const card = retryBtn.closest('.algorithm-detail-card');
                console.log(`재시도 버튼 클릭: ${algorithmId}`);
                this.executeAlgorithm(algorithmId, card);
                return;
            }
        });
    }
    
    createAlgorithmPreviewCard(id, algorithm) {
        const card = document.createElement('div');
        card.className = 'algorithm-preview-card';
        card.innerHTML = `
            <div class="algorithm-icon" style="background: ${algorithm.color || '#667eea'}">
                ${algorithm.icon || '🎯'}
            </div>
            <h4 class="algorithm-name">${algorithm.name || 'Unknown'}</h4>
            <p class="algorithm-subtitle">${algorithm.subtitle || ''}</p>
            <div class="algorithm-accuracy">
                <span class="accuracy-label">정확도</span>
                <span class="accuracy-value">${algorithm.accuracy || 'N/A'}</span>
            </div>
            <button class="quick-run-btn" data-algorithm="${id}">
                <i class="fas fa-play mr-2"></i>빠른 실행
            </button>
        `;
        
        return card;
    }
    
    createAlgorithmDetailCard(id, algorithm) {
        const card = document.createElement('div');
        card.className = 'algorithm-detail-card glass-card';
        card.setAttribute('data-algorithm', id);
        
        card.innerHTML = `
            <div class="card-header">
                <div class="algorithm-icon-large" style="background: ${algorithm.color || '#667eea'}">
                    ${algorithm.icon || '🎯'}
                </div>
                <div class="algorithm-meta">
                    <h3 class="algorithm-title">${algorithm.name || 'Unknown'}</h3>
                    <p class="algorithm-subtitle-large">${algorithm.subtitle || ''}</p>
                    <div class="algorithm-badges">
                        <span class="badge accuracy-badge">${algorithm.accuracy || 'N/A'}</span>
                        <span class="badge complexity-badge">${algorithm.complexity || 'Medium'}</span>
                        <span class="badge time-badge">${algorithm.execution_time || '~2초'}</span>
                    </div>
                </div>
            </div>
            
            <div class="card-body">
                <p class="algorithm-description">${algorithm.description || '설명 없음'}</p>
                
                ${algorithm.features && algorithm.features.length > 0 ? `
                <div class="algorithm-features">
                    <h4>주요 특징</h4>
                    <ul>
                        ${algorithm.features.map(feature => `<li>${feature}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
                
                <div class="algorithm-stats">
                    <div class="stat-item">
                        <i class="fas fa-target"></i>
                        <span>정확도: ${algorithm.accuracy || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-clock"></i>
                        <span>실행시간: ${algorithm.execution_time || '~2초'}</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-cogs"></i>
                        <span>복잡도: ${algorithm.complexity || 'Medium'}</span>
                    </div>
                </div>
            </div>
            
            <div class="card-footer">
                <button class="btn-execute" data-algorithm="${id}">
                    <i class="fas fa-play mr-2"></i>
                    알고리즘 실행
                    <div class="btn-shine"></div>
                </button>
                <button class="btn-info" data-algorithm="${id}">
                    <i class="fas fa-info-circle"></i>
                </button>
            </div>
            
            <div class="execution-result hidden" id="result-${id}">
                <div class="result-header">
                    <h4><i class="fas fa-chart-line mr-2"></i>실행 결과</h4>
                    <button class="result-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="result-content">
                    <!-- 결과가 동적으로 삽입됩니다 -->
                </div>
            </div>
        `;
        
        return card;
    }
    
    // ===== 알고리즘 실행 =====
    async executeAlgorithm(algorithmId, cardElement = null) {
        console.log(`알고리즘 실행 시작: ${algorithmId}`);
        
        await this.ensureInitialized();
        
        const algorithm = this.algorithms[algorithmId];
        if (!algorithm) {
            console.error(`알고리즘을 찾을 수 없음: ${algorithmId}`);
            window.showToast('알고리즘을 찾을 수 없습니다', 'error');
            return null;
        }
        
        const startTime = performance.now();
        
        try {
            this.showExecutionLoading(algorithmId, cardElement);
            window.showLoading(`${algorithm.name} 실행 중...`);
            
            console.log(`API 호출: /api/execute/${algorithmId}`);
            
            const response = await fetch(`${this.baseURL}/execute/${algorithmId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            console.log(`응답 받음: ${response.status} ${response.statusText}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('실행 결과:', result);
            
            if (result.status === 'success') {
                console.log(`실행 성공: ${result.numbers}`);
                this.handleExecutionSuccess(algorithmId, result, cardElement);
                window.showToast(`${algorithm.name} 실행 완료!`, 'success');
                return result;
            } else {
                console.error(`실행 실패: ${result.message}`);
                this.handleExecutionError(algorithmId, result.message, cardElement);
                window.showToast(result.message || '알고리즘 실행 중 오류 발생', 'error');
                return null;
            }
            
        } catch (error) {
            console.error('알고리즘 실행 중 예외 발생:', error);
            console.error('에러 스택:', error.stack);
            this.handleExecutionError(algorithmId, error.message, cardElement);
            window.showToast('네트워크 오류가 발생했습니다', 'error');
            return null;
        } finally {
            window.hideLoading();
            this.hideExecutionLoading(algorithmId, cardElement);
            
            const duration = performance.now() - startTime;
            console.log(`실행 시간: ${duration.toFixed(2)}ms`);
        }
    }
    
    showExecutionLoading(algorithmId, cardElement) {
        if (cardElement) {
            const executeBtn = cardElement.querySelector('.btn-execute');
            if (executeBtn) {
                executeBtn.innerHTML = `
                    <div class="inline-spinner"></div>
                    실행 중...
                `;
                executeBtn.disabled = true;
            }
        }
    }
    
    hideExecutionLoading(algorithmId, cardElement) {
        if (cardElement) {
            const executeBtn = cardElement.querySelector('.btn-execute');
            if (executeBtn) {
                executeBtn.innerHTML = `
                    <i class="fas fa-play mr-2"></i>
                    알고리즘 실행
                    <div class="btn-shine"></div>
                `;
                executeBtn.disabled = false;
            }
        }
    }
    
    handleExecutionSuccess(algorithmId, result, cardElement) {
        const numbers = result.numbers || [];
        const algorithm = this.algorithms[algorithmId];
        
        if (cardElement) {
            this.showResultInCard(algorithmId, {
                success: true,
                numbers: numbers,
                algorithm: algorithm,
                timestamp: result.timestamp,
                cached: result.cached || false
            }, cardElement);
        }
        
        this.updateGlobalResult(algorithmId, numbers, algorithm);
        this.addExecutionHistory({
            algorithmId,
            timestamp: new Date().toISOString(),
            success: true,
            numbers: numbers
        });
    }
    
    handleExecutionError(algorithmId, errorMessage, cardElement) {
        if (cardElement) {
            this.showResultInCard(algorithmId, {
                success: false,
                error: errorMessage
            }, cardElement);
        }
        
        this.addExecutionHistory({
            algorithmId,
            timestamp: new Date().toISOString(),
            success: false,
            error: errorMessage
        });
    }
    
    showResultInCard(algorithmId, result, cardElement) {
        const resultContainer = cardElement.querySelector(`#result-${algorithmId}`);
        if (!resultContainer) {
            console.warn(`결과 컨테이너를 찾을 수 없음: result-${algorithmId}`);
            return;
        }
        
        const resultContent = resultContainer.querySelector('.result-content');
        
        if (result.success) {
            resultContent.innerHTML = `
                <div class="result-numbers">
                    ${result.numbers.map(num => `
                        <div class="result-number" style="background: ${result.algorithm.color || '#667eea'}">
                            ${num}
                        </div>
                    `).join('')}
                </div>
                
                <div class="result-metadata">
                    <div class="meta-item">
                        <i class="fas fa-calendar mr-2"></i>
                        생성 시간: ${new Date(result.timestamp).toLocaleString('ko-KR')}
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-chart-line mr-2"></i>
                        예상 정확도: ${result.algorithm.accuracy || 'N/A'}
                    </div>
                    ${result.cached ? '<div class="meta-item"><i class="fas fa-database mr-2"></i>캐시된 결과</div>' : ''}
                </div>
                
                <div class="result-actions">
                    <button class="btn-save-result" data-numbers="${result.numbers.join(',')}" data-algorithm="${algorithmId}">
                        <i class="fas fa-save mr-2"></i>결과 저장
                    </button>
                    <button class="btn-copy-result" data-numbers="${result.numbers.join(', ')}">
                        <i class="fas fa-copy mr-2"></i>복사
                    </button>
                </div>
            `;
        } else {
            resultContent.innerHTML = `
                <div class="result-error">
                    <i class="fas fa-exclamation-triangle text-red-400 text-2xl mb-3"></i>
                    <h4>실행 오류</h4>
                    <p>${result.error || '알 수 없는 오류가 발생했습니다'}</p>
                    <button class="btn-retry" data-algorithm="${algorithmId}">
                        <i class="fas fa-redo mr-2"></i>다시 시도
                    </button>
                </div>
            `;
        }
        
        resultContainer.classList.remove('hidden');
        this.animateResultShow(resultContainer);
    }
    
    // ===== 유틸리티 함수 =====
    animateCards(cards) {
        Array.from(cards).forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.5s ease-out';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }
    
    animateResultShow(container) {
        container.style.maxHeight = '0';
        container.style.opacity = '0';
        container.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            container.style.transition = 'all 0.3s ease-out';
            container.style.maxHeight = '500px';
            container.style.opacity = '1';
            container.style.transform = 'translateY(0)';
        }, 50);
    }
    
    hideResult(algorithmId) {
        const resultContainer = document.getElementById(`result-${algorithmId}`);
        if (resultContainer) {
            resultContainer.style.opacity = '0';
            resultContainer.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                resultContainer.classList.add('hidden');
                resultContainer.style.maxHeight = '0';
            }, 300);
        }
    }
    
    updateGlobalResult(algorithmId, numbers, algorithm) {
        const globalResultContainer = document.getElementById('recommended-numbers');
        if (globalResultContainer) {
            globalResultContainer.innerHTML = numbers.map(num => `
                <div class="prediction-number">
                    <span>${num}</span>
                </div>
            `).join('');
            
            const predictionResult = document.getElementById('prediction-result');
            if (predictionResult) {
                predictionResult.classList.remove('hidden');
            }
        }
    }
    
    async savePrediction(numbers, algorithmId) {
        console.log('예측 저장 시작:', { numbers, algorithmId });
        
        try {
            const response = await fetch('/api/save-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    numbers: numbers,
                    algorithm: algorithmId,
                    algorithm_name: this.algorithms[algorithmId]?.name || 'Unknown',
                    timestamp: new Date().toISOString()
                })
            });
            
            const result = await response.json();
            console.log('저장 응답:', result);
            
            if (result.status === 'success') {
                window.showToast('예측이 저장되었습니다!', 'success');
                if (window.lottoApp) {
                    window.lottoApp.loadUserData();
                }
            } else {
                window.showToast(result.message || '저장 중 오류가 발생했습니다', 'error');
            }
            
        } catch (error) {
            console.error('예측 저장 실패:', error);
            window.showToast('네트워크 오류가 발생했습니다', 'error');
        }
    }
    
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            window.showToast('클립보드에 복사되었습니다', 'success', 2000);
        } catch (error) {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            document.body.appendChild(textArea);
            textArea.select();
            
            try {
                document.execCommand('copy');
                window.showToast('클립보드에 복사되었습니다', 'success', 2000);
            } catch (err) {
                window.showToast('복사에 실패했습니다', 'error');
            }
            
            document.body.removeChild(textArea);
        }
    }
    
    showAlgorithmInfo(algorithmId, algorithm) {
        console.log('알고리즘 정보 표시:', algorithmId);
        const modal = this.createInfoModal(algorithm);
        document.body.appendChild(modal);
        
        setTimeout(() => {
            modal.classList.add('show');
        }, 10);
    }
    
    createInfoModal(algorithm) {
        const modal = document.createElement('div');
        modal.className = 'info-modal-overlay';
        modal.innerHTML = `
            <div class="info-modal">
                <div class="modal-header">
                    <h2>${algorithm.name || 'Unknown'}</h2>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="algorithm-detail-info">
                        <h3>알고리즘 상세 정보</h3>
                        <p>${algorithm.description || '설명 없음'}</p>
                        
                        <div class="info-grid">
                            <div class="info-item">
                                <strong>정확도:</strong> ${algorithm.accuracy || 'N/A'}
                            </div>
                            <div class="info-item">
                                <strong>복잡도:</strong> ${algorithm.complexity || 'Medium'}
                            </div>
                            <div class="info-item">
                                <strong>실행시간:</strong> ${algorithm.execution_time || '~2초'}
                            </div>
                            <div class="info-item">
                                <strong>버전:</strong> ${algorithm.version || '1.0'}
                            </div>
                        </div>
                        
                        ${algorithm.features && algorithm.features.length > 0 ? `
                        <div class="features-list">
                            <h4>주요 특징</h4>
                            <ul>
                                ${algorithm.features.map(feature => `<li>${feature}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
        
        const closeBtn = modal.querySelector('.modal-close');
        const overlay = modal;
        
        closeBtn.addEventListener('click', () => this.closeModal(modal));
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.closeModal(modal);
            }
        });
        
        return modal;
    }
    
    closeModal(modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            modal.remove();
        }, 300);
    }
    
    addExecutionHistory(record) {
        this.executionHistory.unshift(record);
        
        if (this.executionHistory.length > 100) {
            this.executionHistory.pop();
        }
        
        try {
            localStorage.setItem('lotto_execution_history', JSON.stringify(this.executionHistory.slice(0, 50)));
        } catch (e) {
            console.warn('localStorage 저장 실패:', e);
        }
    }
}

// ===== 전역 인스턴스 및 함수 =====
let githubManager = null;

async function loadAlgorithmPreview() {
    console.log('loadAlgorithmPreview 호출됨');
    if (!githubManager) {
        githubManager = new GitHubAPIManager();
        await githubManager.init();
    }
}

async function executeAlgorithm(algorithmId) {
    console.log(`executeAlgorithm 호출됨: ${algorithmId}`);
    if (!githubManager) {
        githubManager = new GitHubAPIManager();
        await githubManager.init();
    }
    
    return await githubManager.executeAlgorithm(algorithmId);
}

// ===== DOMContentLoaded 초기화 =====
document.addEventListener('DOMContentLoaded', async function() {
    console.log('DOMContentLoaded - GitHubAPIManager 초기화 시작');
    
    try {
        githubManager = new GitHubAPIManager();
        await githubManager.init();
        
        // 전역 접근
        window.githubManager = githubManager;
        window.executeAlgorithm = executeAlgorithm;
        window.loadAlgorithmPreview = loadAlgorithmPreview;
        
        console.log('GitHubAPIManager 전역 등록 완료');
        console.log('상태:', {
            initialized: githubManager.isInitialized,
            algorithmCount: Object.keys(githubManager.algorithms).length
        });
        
    } catch (error) {
        console.error('GitHubAPIManager 초기화 중 오류:', error);
    }
});

console.log('github-api.js 로드 완료');
