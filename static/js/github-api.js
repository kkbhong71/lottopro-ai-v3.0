/**
 * GitHub API 관리자 - LottoPro-AI v3.0
 * 최적화 버전: 중복 제거 + 캐싱 + 폴백 + 강화된 에러 처리
 */

class GitHubAPIManager {
    constructor() {
        this.baseUrl = '/api';
        this.algorithms = {};
        this.initialized = false;
        
        // 캐싱 시스템
        this.executionCache = new Map();
        this.cacheTimeout = 3600000; // 1시간 (밀리초)
        
        console.log('🔧 GitHubAPIManager 생성됨');
    }

    /**
     * 초기화 함수
     */
    async initialize() {
        if (this.initialized) {
            console.log('✅ 이미 초기화됨');
            return;
        }

        try {
            console.log('📥 알고리즘 정보 로드 시작...');
            await this.loadAlgorithmInfo();
            this.initialized = true;
            console.log('✅ GitHubAPIManager 초기화 완료');
        } catch (error) {
            console.error('❌ 초기화 실패:', error);
            // 초기화 실패해도 폴백 알고리즘으로 작동
            this.algorithms = this.getDefaultAlgorithms();
            this.initialized = true;
        }
    }

    /**
     * 알고리즘 정보 로드 (폴백 포함)
     */
    async loadAlgorithmInfo() {
        try {
            const url = `${this.baseUrl}/algorithm-info`;
            console.log('🌐 API 호출:', url);
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 API 응답 받음:', data);
            
            if (data.status === 'success' && data.info) {
                this.algorithms = data.info;
                console.log(`✅ 알고리즘 ${Object.keys(data.info).length}개 로드 완료`);
                console.log('📋 로드된 알고리즘:', Object.keys(data.info));
            } else {
                throw new Error('Invalid response format');
            }
            
        } catch (error) {
            console.error('❌ 알고리즘 정보 로드 실패:', error);
            
            // 폴백: 기본 알고리즘 사용
            console.log('⚠️ 폴백 알고리즘 사용');
            this.algorithms = this.getDefaultAlgorithms();
            
            // 사용자에게 알림
            if (window.showToast) {
                window.showToast('알고리즘 정보를 불러오는데 실패했습니다. 기본 알고리즘을 사용합니다.', 'warning');
            }
        }
    }

    /**
     * 기본 알고리즘 (폴백용)
     */
    getDefaultAlgorithms() {
        return {
            'super_v1': {
                name: 'Super ver 1.0',
                description: 'Feature Engineering Focused - 고급 특성 공학 기반',
                accuracy: '87.3',
                execution_time: '~2초',
                difficulty: 'hard',
                github_path: 'algorithms/super_v1.py'
            },
            'ultimate_v3': {
                name: 'Ultimate Prediction 3.0',
                description: 'AI 심층 예측 시스템',
                accuracy: '82.1',
                execution_time: '~2초',
                difficulty: 'hard',
                github_path: 'algorithms/ultimate_v3.py'
            },
            'ultimate_v2': {
                name: 'Ultimate Prediction 2.0',
                description: 'Advanced Pattern Recognition',
                accuracy: '78.8',
                execution_time: '~1.5초',
                difficulty: 'medium',
                github_path: 'algorithms/ultimate_v2.py'
            },
            'ultimate_v1': {
                name: 'Ultimate Prediction 1.0',
                description: 'Basic Statistical Analysis',
                accuracy: '73.2',
                execution_time: '~1초',
                difficulty: 'easy',
                github_path: 'algorithms/ultimate_v1.py'
            }
        };
    }

    /**
     * 알고리즘 카드 렌더링 (중복 제거 + 정렬/필터)
     */
    renderAlgorithmCards(containerId = 'algorithm-grid', sortBy = 'accuracy', filterAccuracy = 'all') {
        const container = document.getElementById(containerId);
        
        if (!container) {
            console.error(`❌ 컨테이너를 찾을 수 없음: ${containerId}`);
            return;
        }

        console.log('🎨 카드 렌더링 시작:', { sortBy, filterAccuracy });

        // 🔥 중복 제거를 위해 Set 사용
        const algorithmKeys = [...new Set(Object.keys(this.algorithms))];
        
        let filteredAlgorithms = algorithmKeys.map(key => ({
            key: key,
            ...this.algorithms[key]
        }));

        console.log(`📊 총 ${filteredAlgorithms.length}개 알고리즘 (중복 제거 후)`);

        // 필터 적용
        if (filterAccuracy !== 'all') {
            const beforeFilter = filteredAlgorithms.length;
            
            filteredAlgorithms = filteredAlgorithms.filter(algo => {
                const accuracy = parseFloat(algo.accuracy) || 0;
                
                switch(filterAccuracy) {
                    case 'high':
                        return accuracy >= 85;
                    case 'medium':
                        return accuracy >= 70 && accuracy < 85;
                    case 'low':
                        return accuracy < 70;
                    default:
                        return true;
                }
            });
            
            console.log(`🔍 필터 적용: ${beforeFilter}개 → ${filteredAlgorithms.length}개`);
        }

        // 정렬 적용
        filteredAlgorithms.sort((a, b) => {
            if (sortBy === 'accuracy') {
                return (parseFloat(b.accuracy) || 0) - (parseFloat(a.accuracy) || 0);
            } else if (sortBy === 'name') {
                return (a.name || '').localeCompare(b.name || '');
            }
            return 0;
        });

        console.log(`📋 정렬 기준: ${sortBy}`);

        // 컨테이너 비우기
        container.innerHTML = '';

        // 알고리즘이 없는 경우
        if (filteredAlgorithms.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1; text-align: center; padding: 3rem;">
                    <i class="fas fa-inbox" style="font-size: 4rem; color: #666; margin-bottom: 1rem;"></i>
                    <p style="font-size: 1.2rem; color: #999;">조건에 맞는 알고리즘이 없습니다</p>
                </div>
            `;
            return;
        }

        // 카드 렌더링
        filteredAlgorithms.forEach(algo => {
            const card = this.createAlgorithmCard(algo);
            container.appendChild(card);
        });

        // 🔥 렌더링 후 이벤트 리스너 재바인딩
        this.bindCardEvents();

        console.log(`✅ ${filteredAlgorithms.length}개 카드 렌더링 완료`);
    }

    /**
     * 알고리즘 카드 생성
     */
    createAlgorithmCard(algo) {
        const card = document.createElement('div');
        card.className = 'algorithm-card';
        card.setAttribute('data-algorithm', algo.key);
        
        const accuracy = parseFloat(algo.accuracy) || 0;
        const accuracyClass = accuracy >= 85 ? 'high' : accuracy >= 70 ? 'medium' : 'low';
        const isVerified = accuracy >= 80;

        card.innerHTML = `
            <div class="algorithm-header">
                <div class="algorithm-title-section">
                    <h3>${algo.name || '알고리즘'}</h3>
                    ${isVerified ? '<span class="verified-badge">✓ 검증됨</span>' : ''}
                </div>
                <span class="accuracy-badge ${accuracyClass}">
                    ${accuracy.toFixed(1)}%
                </span>
            </div>
            <p class="algorithm-description">${algo.description || '설명 없음'}</p>
            <div class="algorithm-stats">
                <div class="stat">
                    <i class="fas fa-clock"></i>
                    <span>${algo.execution_time || '~2초'}</span>
                </div>
                <div class="stat">
                    <i class="fas fa-layer-group"></i>
                    <span>${algo.difficulty || 'medium'}</span>
                </div>
            </div>
            <button class="run-algorithm-btn" data-algorithm="${algo.key}">
                <span class="btn-icon">🚀</span>
                <span class="btn-text">AI 예측 실행</span>
            </button>
        `;

        return card;
    }

    /**
     * 카드 이벤트 바인딩 (중복 방지)
     */
    bindCardEvents() {
        const buttons = document.querySelectorAll('.run-algorithm-btn');
        console.log(`🔗 ${buttons.length}개 버튼에 이벤트 바인딩 시작`);

        buttons.forEach(button => {
            // 🔥 기존 이벤트 리스너 제거 (중복 방지)
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);

            // 새로운 이벤트 리스너 추가
            newButton.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                const algorithmKey = newButton.getAttribute('data-algorithm');
                console.log('🎯 알고리즘 실행 버튼 클릭:', algorithmKey);
                
                await this.runAlgorithm(algorithmKey);
            });
        });

        console.log('✅ 이벤트 바인딩 완료');
    }

    /**
     * 캐시된 결과 확인
     */
    getCachedResult(algorithmKey) {
        const cached = this.executionCache.get(algorithmKey);
        
        if (!cached) {
            console.log('💾 캐시 없음:', algorithmKey);
            return null;
        }
        
        const now = Date.now();
        const age = now - cached.timestamp;
        
        if (age > this.cacheTimeout) {
            console.log('⏰ 캐시 만료됨:', algorithmKey, `(${Math.round(age/1000/60)}분 경과)`);
            this.executionCache.delete(algorithmKey);
            return null;
        }
        
        console.log('💾 캐시 사용:', algorithmKey, `(${Math.round(age/1000/60)}분 전)`);
        return {
            ...cached.result,
            cached: true
        };
    }

    /**
     * 결과 캐싱
     */
    cacheResult(algorithmKey, result) {
        this.executionCache.set(algorithmKey, {
            result: result,
            timestamp: Date.now()
        });
        console.log('💾 결과 캐시됨:', algorithmKey);
    }

    /**
     * 알고리즘 실행 (캐싱 + 강화된 에러 처리)
     */
    async runAlgorithm(algorithmKey) {
        const button = document.querySelector(`button[data-algorithm="${algorithmKey}"]`);
        
        try {
            // 버튼 상태 변경
            if (button) {
                button.disabled = true;
                button.innerHTML = `
                    <span class="btn-icon">⏳</span>
                    <span class="btn-text">실행 중...</span>
                `;
            }

            console.log('🚀 알고리즘 실행 시작:', algorithmKey);

            // 🔥 캐시 확인
            const cachedResult = this.getCachedResult(algorithmKey);
            if (cachedResult) {
                this.displayPredictionResult(cachedResult);
                
                // 버튼 복원
                if (button) {
                    button.disabled = false;
                    button.innerHTML = `
                        <span class="btn-icon">🚀</span>
                        <span class="btn-text">AI 예측 실행</span>
                    `;
                }
                
                return;
            }

            // API 호출
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    algorithm: algorithmKey,
                    user_numbers: []
                })
            });

            // 에러 응답 처리
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status} 에러`);
            }

            const result = await response.json();
            console.log('📊 예측 결과:', result);

            // 결과 상태 체크
            if (result.status === 'error') {
                throw new Error(result.message || '알고리즘 실행 실패');
            }

            // 빈 결과 체크
            if (!result.numbers || result.numbers.length === 0) {
                throw new Error('예측 번호가 생성되지 않았습니다. 잠시 후 다시 시도해주세요.');
            }

            // 번호 유효성 검증
            if (result.numbers.length !== 6) {
                throw new Error(`잘못된 번호 개수: ${result.numbers.length}개 (6개 필요)`);
            }

            // 🔥 결과 캐싱
            this.cacheResult(algorithmKey, result);

            // 결과 표시
            this.displayPredictionResult(result);

            // 성공 토스트
            if (window.showToast) {
                window.showToast('AI 예측 완료!', 'success');
            }

        } catch (error) {
            console.error('❌ 알고리즘 실행 실패:', error);
            
            // 사용자 친화적 에러 메시지
            let errorMessage = error.message;
            
            // 특정 에러에 대한 친화적 메시지
            if (errorMessage.includes('warnings')) {
                errorMessage = '서버 설정 문제가 발생했습니다. 잠시 후 다시 시도해주세요.';
            } else if (errorMessage.includes('timeout')) {
                errorMessage = '서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.';
            } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
                errorMessage = '네트워크 연결을 확인해주세요.';
            }
            
            // 에러 토스트
            if (window.showToast) {
                window.showToast(`알고리즘 실행 실패: ${errorMessage}`, 'error', 5000);
            } else {
                alert(`알고리즘 실행 실패\n\n${errorMessage}`);
            }
            
        } finally {
            // 버튼 복원
            if (button) {
                button.disabled = false;
                button.innerHTML = `
                    <span class="btn-icon">🚀</span>
                    <span class="btn-text">AI 예측 실행</span>
                `;
            }
        }
    }

    /**
     * 예측 결과 표시 (모달 방식)
     */
    displayPredictionResult(result) {
        // 기존 모달 제거
        const existingModal = document.getElementById('prediction-result');
        if (existingModal) {
            existingModal.remove();
        }

        // 새 모달 생성
        const resultContainer = document.createElement('div');
        resultContainer.id = 'prediction-result';
        resultContainer.className = 'prediction-result-modal';

        const numbers = result.numbers || [];
        const numbersHtml = numbers.map((num, index) => 
            `<span class="lotto-number" style="animation-delay: ${index * 0.1}s">${num}</span>`
        ).join('');
        
        // 알고리즘 정보
        const algorithmInfo = this.algorithms[result.algorithm] || {};
        const algorithmName = result.algorithm_name || algorithmInfo.name || result.algorithm;
        const accuracy = result.accuracy_rate || algorithmInfo.accuracy || 'N/A';

        resultContainer.innerHTML = `
            <div class="modal-overlay" onclick="this.closest('.prediction-result-modal').remove()"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h2>🎯 AI 예측 결과</h2>
                    <button class="close-modal" onclick="this.closest('.prediction-result-modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="predicted-numbers">
                        ${numbersHtml}
                    </div>
                    <div class="result-info">
                        <div class="info-item">
                            <i class="fas fa-robot"></i>
                            <span><strong>알고리즘:</strong> ${algorithmName}</span>
                        </div>
                        <div class="info-item">
                            <i class="fas fa-chart-line"></i>
                            <span><strong>예측 정확도:</strong> ${accuracy}%</span>
                        </div>
                        <div class="info-item">
                            <i class="fas fa-clock"></i>
                            <span><strong>예측 시간:</strong> ${new Date().toLocaleString('ko-KR')}</span>
                        </div>
                        ${result.cached ? '<div class="cache-info"><i class="fas fa-bolt"></i> 캐시된 결과 (1시간 이내)</div>' : ''}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-save" onclick="window.saveNumbers && window.saveNumbers(${JSON.stringify(numbers)}, '${algorithmName}'); window.showToast && window.showToast('번호가 저장되었습니다!', 'success');">
                        <i class="fas fa-save"></i> 번호 저장하기
                    </button>
                    <button class="btn-close" onclick="this.closest('.prediction-result-modal').remove()">
                        <i class="fas fa-times-circle"></i> 닫기
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(resultContainer);

        // 애니메이션을 위한 약간의 지연
        setTimeout(() => {
            resultContainer.classList.add('show');
        }, 10);

        console.log('✅ 결과 모달 표시됨');
    }
}

// ===== 전역 초기화 =====

let githubAPIManager = null;

// DOM 로드 완료 후 초기화
document.addEventListener('DOMContentLoaded', async () => {
    console.log('📄 DOMContentLoaded - GitHubAPIManager 초기화 시작');
    
    try {
        githubAPIManager = new GitHubAPIManager();
        await githubAPIManager.initialize();
        
        // 알고리즘 페이지인 경우 자동 렌더링
        if (document.getElementById('algorithm-grid')) {
            console.log('🎨 알고리즘 페이지 감지 - 카드 렌더링 시작');
            githubAPIManager.renderAlgorithmCards('algorithm-grid', 'accuracy', 'all');
        }
        
        // 홈페이지 미리보기인 경우
        if (document.getElementById('algorithm-preview')) {
            console.log('🏠 홈페이지 미리보기 감지 - 카드 렌더링 시작');
            githubAPIManager.renderAlgorithmCards('algorithm-preview', 'accuracy', 'all');
        }
        
        // 전역 객체로 등록
        window.githubAPIManager = githubAPIManager;
        window.githubManager = githubAPIManager; // 별칭
        
        console.log('✅ GitHubAPIManager 전역 등록 완료');
        console.log('📊 상태:', {
            initialized: githubAPIManager.initialized,
            algorithmCount: Object.keys(githubAPIManager.algorithms).length,
            cachedResults: githubAPIManager.executionCache.size
        });
        
    } catch (error) {
        console.error('❌ GitHubAPIManager 초기화 중 오류:', error);
    }
});

// ===== 전역 함수 (하위 호환성) =====

/**
 * 알고리즘 미리보기 로드 (홈페이지용)
 */
async function loadAlgorithmPreview() {
    if (!githubAPIManager) {
        console.warn('⚠️ GitHubAPIManager가 아직 초기화되지 않았습니다');
        return;
    }
    
    try {
        console.log('🏠 알고리즘 미리보기 로드 시작');
        
        if (!githubAPIManager.initialized) {
            await githubAPIManager.initialize();
        }
        
        const previewContainer = document.getElementById('algorithm-preview');
        if (previewContainer) {
            githubAPIManager.renderAlgorithmCards('algorithm-preview', 'accuracy', 'all');
        }
    } catch (error) {
        console.error('❌ 알고리즘 미리보기 로드 실패:', error);
    }
}

// 전역 함수로 노출
window.loadAlgorithmPreview = loadAlgorithmPreview;

console.log('📦 github-api.js 로드 완료');

// ===== 번호 저장 함수 (localStorage) =====

/**
 * 예측된 번호를 localStorage에 저장
 * @param {Array} numbers - 6개의 로또 번호
 * @param {String} algorithmName - 알고리즘 이름
 */
window.saveNumbers = function(numbers, algorithmName) {
    console.log('💾 번호 저장:', numbers, algorithmName);
    
    try {
        // 유효성 검사
        if (!Array.isArray(numbers) || numbers.length !== 6) {
            throw new Error('잘못된 번호 형식');
        }
        
        if (!numbers.every(n => Number.isInteger(n) && n >= 1 && n <= 45)) {
            throw new Error('번호는 1-45 사이의 정수여야 합니다');
        }
        
        // 기존 데이터 로드
        const savedNumbers = JSON.parse(localStorage.getItem('savedNumbers') || '[]');
        
        // 새 항목 생성 (saved_numbers.html 형식)
        const newEntry = {
            id: Date.now(),
            numbers: numbers,
            timestamp: new Date().toISOString(),
            algorithm: algorithmName || 'AI 예측',
            checked: false,
            matches: 0
        };
        
        // 맨 앞에 추가 (최신순)
        savedNumbers.unshift(newEntry);
        
        // 최대 100개까지만 유지
        const trimmed = savedNumbers.slice(0, 100);
        
        // 저장
        localStorage.setItem('savedNumbers', JSON.stringify(trimmed));
        
        console.log('✅ 저장 완료. 총', trimmed.length, '개');
        
        return true;
        
    } catch (error) {
        console.error('❌ 저장 실패:', error);
        
        if (window.showToast) {
            window.showToast('번호 저장에 실패했습니다: ' + error.message, 'error');
        } else {
            alert('저장 실패: ' + error.message);
        }
        
        return false;
    }
};

console.log('💾 saveNumbers 함수 등록 완료');
