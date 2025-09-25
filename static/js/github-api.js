/**
 * GitHub API 연동 모듈
 * 알고리즘 코드 실행 및 관리
 */

class GitHubAPIManager {
    constructor() {
        this.baseURL = '/api';
        this.algorithms = {};
        this.executionHistory = [];
        this.init();
    }
    
    async init() {
        await this.loadAlgorithmInfo();
        this.setupAlgorithmCards();
    }
    
    // ===== 알고리즘 정보 로드 =====
    async loadAlgorithmInfo() {
        try {
            const response = await fetch('/static/algorithms/algorithm_info.json');
            if (!response.ok) {
                // 서버에서 동적으로 로드
                const apiResponse = await fetch('/api/algorithm-info/all');
                const data = await apiResponse.json();
                this.algorithms = data.algorithms || {};
            } else {
                const data = await response.json();
                this.algorithms = data.algorithms || {};
            }
            
            console.log('알고리즘 정보 로드 완료:', Object.keys(this.algorithms).length + '개');
        } catch (error) {
            console.error('알고리즘 정보 로드 실패:', error);
            this.algorithms = this.getFallbackAlgorithms();
        }
    }
    
    getFallbackAlgorithms() {
        return {
            super_v1: {
                name: "Super ver 1.0",
                subtitle: "Feature Engineering Focused",
                description: "고급 피처 엔지니어링 기법을 활용한 예측 모델",
                accuracy: "78.5%",
                icon: "🔧",
                color: "#FF6B6B"
            },
            strongest_universe_v1: {
                name: "The Strongest in Universe",
                subtitle: "Maximum Power Algorithm", 
                description: "우주 최강의 예측 알고리즘",
                accuracy: "82.3%",
                icon: "💪",
                color: "#4ECDC4"
            }
        };
    }
    
    // ===== 알고리즘 카드 설정 =====
    setupAlgorithmCards() {
        const previewContainer = document.getElementById('algorithm-preview');
        const algorithmGrid = document.getElementById('algorithm-grid');
        
        if (previewContainer) {
            this.renderAlgorithmPreview(previewContainer);
        }
        
        if (algorithmGrid) {
            this.renderAlgorithmGrid(algorithmGrid);
        }
    }
    
    renderAlgorithmPreview(container) {
        container.innerHTML = '';
        
        // 상위 4개 알고리즘만 미리보기로 표시
        const topAlgorithms = Object.entries(this.algorithms).slice(0, 4);
        
        topAlgorithms.forEach(([id, algorithm]) => {
            const card = this.createAlgorithmPreviewCard(id, algorithm);
            container.appendChild(card);
        });
        
        // 애니메이션 효과
        this.animateCards(container.children);
    }
    
    renderAlgorithmGrid(container) {
        container.innerHTML = '';
        
        Object.entries(this.algorithms).forEach(([id, algorithm]) => {
            const card = this.createAlgorithmDetailCard(id, algorithm);
            container.appendChild(card);
        });
        
        // 애니메이션 효과
        this.animateCards(container.children);
    }
    
    createAlgorithmPreviewCard(id, algorithm) {
        const card = document.createElement('div');
        card.className = 'algorithm-preview-card';
        card.innerHTML = `
            <div class="algorithm-icon" style="background: ${algorithm.color}">
                ${algorithm.icon}
            </div>
            <h4 class="algorithm-name">${algorithm.name}</h4>
            <p class="algorithm-subtitle">${algorithm.subtitle}</p>
            <div class="algorithm-accuracy">
                <span class="accuracy-label">정확도</span>
                <span class="accuracy-value">${algorithm.accuracy}</span>
            </div>
            <button class="quick-run-btn" data-algorithm="${id}">
                <i class="fas fa-play mr-2"></i>빠른 실행
            </button>
        `;
        
        // 이벤트 리스너 추가
        card.querySelector('.quick-run-btn').addEventListener('click', (e) => {
            e.preventDefault();
            this.executeAlgorithm(id);
        });
        
        return card;
    }
    
    createAlgorithmDetailCard(id, algorithm) {
        const card = document.createElement('div');
        card.className = 'algorithm-detail-card glass-card';
        card.innerHTML = `
            <div class="card-header">
                <div class="algorithm-icon-large" style="background: ${algorithm.color}">
                    ${algorithm.icon}
                </div>
                <div class="algorithm-meta">
                    <h3 class="algorithm-title">${algorithm.name}</h3>
                    <p class="algorithm-subtitle-large">${algorithm.subtitle}</p>
                    <div class="algorithm-badges">
                        <span class="badge accuracy-badge">${algorithm.accuracy}</span>
                        <span class="badge complexity-badge">${algorithm.complexity || 'Medium'}</span>
                        <span class="badge time-badge">${algorithm.execution_time || '~2초'}</span>
                    </div>
                </div>
            </div>
            
            <div class="card-body">
                <p class="algorithm-description">${algorithm.description}</p>
                
                <div class="algorithm-features">
                    <h4>주요 특징</h4>
                    <ul>
                        ${(algorithm.features || []).map(feature => `<li>${feature}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="algorithm-stats">
                    <div class="stat-item">
                        <i class="fas fa-target"></i>
                        <span>정확도: ${algorithm.accuracy}</span>
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
        
        // 이벤트 리스너 추가
        const executeBtn = card.querySelector('.btn-execute');
        const infoBtn = card.querySelector('.btn-info');
        const closeBtn = card.querySelector('.result-close');
        
        executeBtn.addEventListener('click', () => this.executeAlgorithm(id, card));
        infoBtn.addEventListener('click', () => this.showAlgorithmInfo(id, algorithm));
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideResult(id));
        }
        
        return card;
    }
    
    // ===== 알고리즘 실행 =====
    async executeAlgorithm(algorithmId, cardElement = null) {
        const algorithm = this.algorithms[algorithmId];
        if (!algorithm) {
            window.showToast('알고리즘을 찾을 수 없습니다', 'error');
            return;
        }
        
        const startTime = performance.now();
        
        try {
            // 로딩 표시
            this.showExecutionLoading(algorithmId, cardElement);
            window.showLoading(`${algorithm.name} 실행 중...`);
            
            // API 호출
            const response = await fetch(`${this.baseURL}/execute/${algorithmId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.handleExecutionSuccess(algorithmId, result, cardElement);
                window.showToast(`${algorithm.name} 실행 완료!`, 'success');
            } else {
                this.handleExecutionError(algorithmId, result.message, cardElement);
                window.showToast('알고리즘 실행 중 오류 발생', 'error');
            }
            
        } catch (error) {
            console.error('알고리즘 실행 실패:', error);
            this.handleExecutionError(algorithmId, error.message, cardElement);
            window.showToast('네트워크 오류가 발생했습니다', 'error');
        } finally {
            window.hideLoading();
            this.hideExecutionLoading(algorithmId, cardElement);
            
            // 성능 로깅
            const duration = performance.now() - startTime;
            console.log(`⚡ ${algorithm.name} 실행 완료: ${duration.toFixed(2)}ms`);
        }
        
        // 실행 기록 저장
        this.addExecutionHistory({
            algorithmId,
            timestamp: new Date().toISOString(),
            success: true
        });
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
        
        // 결과 표시
        if (cardElement) {
            this.showResultInCard(algorithmId, {
                success: true,
                numbers: numbers,
                algorithm: algorithm,
                timestamp: result.timestamp
            }, cardElement);
        }
        
        // 전역 결과 업데이트
        this.updateGlobalResult(algorithmId, numbers, algorithm);
        
        // 자동 저장 옵션 표시
        this.showSaveOption(numbers, algorithmId);
    }
    
    handleExecutionError(algorithmId, errorMessage, cardElement) {
        if (cardElement) {
            this.showResultInCard(algorithmId, {
                success: false,
                error: errorMessage
            }, cardElement);
        }
    }
    
    showResultInCard(algorithmId, result, cardElement) {
        const resultContainer = cardElement.querySelector(`#result-${algorithmId}`);
        if (!resultContainer) return;
        
        const resultContent = resultContainer.querySelector('.result-content');
        
        if (result.success) {
            resultContent.innerHTML = `
                <div class="result-numbers">
                    ${result.numbers.map(num => `
                        <div class="result-number" style="background: ${result.algorithm.color}">
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
                        예상 정확도: ${result.algorithm.accuracy}
                    </div>
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
            
            // 결과 액션 버튼 이벤트
            const saveBtn = resultContent.querySelector('.btn-save-result');
            const copyBtn = resultContent.querySelector('.btn-copy-result');
            
            if (saveBtn) {
                saveBtn.addEventListener('click', () => this.savePrediction(result.numbers, algorithmId));
            }
            
            if (copyBtn) {
                copyBtn.addEventListener('click', () => this.copyToClipboard(result.numbers.join(', ')));
            }
            
        } else {
            resultContent.innerHTML = `
                <div class="result-error">
                    <i class="fas fa-exclamation-triangle text-red-400 text-2xl mb-3"></i>
                    <h4>실행 오류</h4>
                    <p>${result.error}</p>
                    <button class="btn-retry" data-algorithm="${algorithmId}">
                        <i class="fas fa-redo mr-2"></i>다시 시도
                    </button>
                </div>
            `;
            
            const retryBtn = resultContent.querySelector('.btn-retry');
            if (retryBtn) {
                retryBtn.addEventListener('click', () => this.executeAlgorithm(algorithmId, cardElement));
            }
        }
        
        // 결과 컨테이너 표시
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
        // 전역 결과 업데이트 (예: 메인 페이지의 추천 번호)
        const globalResultContainer = document.getElementById('recommended-numbers');
        if (globalResultContainer) {
            globalResultContainer.innerHTML = numbers.map(num => `
                <div class="prediction-number">
                    <span>${num}</span>
                </div>
            `).join('');
            
            // 결과 영역 표시
            const predictionResult = document.getElementById('prediction-result');
            if (predictionResult) {
                predictionResult.classList.remove('hidden');
            }
        }
    }
    
    showSaveOption(numbers, algorithmId) {
        const saveBtn = document.getElementById('save-prediction');
        if (saveBtn) {
            saveBtn.onclick = () => this.savePrediction(numbers, algorithmId);
        }
    }
    
    async savePrediction(numbers, algorithmId) {
        try {
            const response = await fetch('/api/save-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    numbers: numbers,
                    algorithm: algorithmId,
                    timestamp: new Date().toISOString()
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                window.showToast('예측이 저장되었습니다!', 'success');
                // 통계 업데이트
                if (window.lottoApp) {
                    window.lottoApp.loadUserData();
                }
            } else {
                window.showToast('저장 중 오류가 발생했습니다', 'error');
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
            // 폴백: 텍스트 선택
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            window.showToast('클립보드에 복사되었습니다', 'success', 2000);
        }
    }
    
    showAlgorithmInfo(algorithmId, algorithm) {
        // 모달이나 사이드패널로 상세 정보 표시
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
                    <h2>${algorithm.name}</h2>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="algorithm-detail-info">
                        <h3>알고리즘 상세 정보</h3>
                        <p>${algorithm.description}</p>
                        
                        <div class="info-grid">
                            <div class="info-item">
                                <strong>정확도:</strong> ${algorithm.accuracy}
                            </div>
                            <div class="info-item">
                                <strong>복잡도:</strong> ${algorithm.complexity || 'Medium'}
                            </div>
                            <div class="info-item">
                                <strong>실행시간:</strong> ${algorithm.execution_time || '~2초'}
                            </div>
                        </div>
                        
                        ${algorithm.features ? `
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
        
        // 모달 닫기 이벤트
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
        
        // 최대 100개 기록만 유지
        if (this.executionHistory.length > 100) {
            this.executionHistory.pop();
        }
        
        // 로컬 스토리지에 저장
        localStorage.setItem('lotto_execution_history', JSON.stringify(this.executionHistory.slice(0, 50)));
    }
    
    getExecutionStats() {
        return {
            totalExecutions: this.executionHistory.length,
            successRate: this.executionHistory.filter(record => record.success).length / this.executionHistory.length,
            mostUsedAlgorithm: this.getMostUsedAlgorithm(),
            recentExecutions: this.executionHistory.slice(0, 10)
        };
    }
    
    getMostUsedAlgorithm() {
        const counts = {};
        this.executionHistory.forEach(record => {
            counts[record.algorithmId] = (counts[record.algorithmId] || 0) + 1;
        });
        
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b, '');
    }
}

// ===== 전역 인스턴스 =====
let githubManager;

// ===== 전역 함수들 =====
async function loadAlgorithmPreview() {
    if (!githubManager) {
        githubManager = new GitHubAPIManager();
    }
}

async function executeAlgorithm(algorithmId) {
    if (!githubManager) {
        githubManager = new GitHubAPIManager();
    }
    
    return await githubManager.executeAlgorithm(algorithmId);
}

// ===== 초기화 =====
document.addEventListener('DOMContentLoaded', function() {
    if (!githubManager) {
        githubManager = new GitHubAPIManager();
    }
    
    // 전역 접근을 위해 window에 할당
    window.githubManager = githubManager;
    window.executeAlgorithm = executeAlgorithm;
    window.loadAlgorithmPreview = loadAlgorithmPreview;
});
