/**
 * 예측 관련 JavaScript 모듈
 * 빠른 예측, 번호 관리, 결과 표시 등
 */

class PredictionManager {
    constructor() {
        this.preferredNumbers = [];
        this.lastPrediction = null;
        this.predictionHistory = [];
        this.init();
    }
    
    init() {
        this.setupNumberInputs();
        this.setupPredictionButtons();
        this.loadPredictionHistory();
        console.log('🎲 예측 관리자 초기화 완료');
    }
    
    // ===== 번호 입력 설정 =====
    setupNumberInputs() {
        const numberInputs = document.querySelectorAll('.number-input');
        
        numberInputs.forEach((input, index) => {
            // 입력 제한 및 검증
            input.addEventListener('input', (e) => {
                let value = parseInt(e.target.value);
                
                // 범위 검증
                if (value < 1 || value > 45) {
                    e.target.classList.add('input-error');
                    this.showInputError(e.target, '1-45 범위의 숫자를 입력해주세요');
                } else {
                    e.target.classList.remove('input-error');
                    this.hideInputError(e.target);
                }
                
                // 중복 검증
                this.checkDuplicateNumbers();
                
                // 다음 입력으로 자동 이동
                if (value && value >= 1 && value <= 45 && index < numberInputs.length - 1) {
                    setTimeout(() => {
                        numberInputs[index + 1].focus();
                    }, 100);
                }
                
                // 현재 선호 번호 업데이트
                this.updatePreferredNumbers();
            });
            
            // Enter 키 처리
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    if (index < numberInputs.length - 1) {
                        numberInputs[index + 1].focus();
                    } else {
                        this.generatePrediction();
                    }
                }
            });
            
            // 붙여넣기 처리
            input.addEventListener('paste', (e) => {
                e.preventDefault();
                const pasteData = e.clipboardData.getData('text');
                this.handleNumberPaste(pasteData, index);
            });
            
            // 포커스 효과
            input.addEventListener('focus', (e) => {
                e.target.parentElement.classList.add('input-focused');
            });
            
            input.addEventListener('blur', (e) => {
                e.target.parentElement.classList.remove('input-focused');
            });
        });
    }
    
    setupPredictionButtons() {
        // 번호 초기화 버튼
        const clearBtn = document.getElementById('clear-numbers');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearNumbers());
        }
        
        // 랜덤 채우기 버튼
        const randomBtn = document.getElementById('random-fill');
        if (randomBtn) {
            randomBtn.addEventListener('click', () => this.fillRandomNumbers());
        }
        
        // 예측 생성 버튼
        const generateBtn = document.getElementById('generate-prediction');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generatePrediction());
        }
        
        // 빠른 예측 버튼
        const quickBtn = document.getElementById('quick-prediction');
        if (quickBtn) {
            quickBtn.addEventListener('click', () => this.quickPrediction());
        }
        
        // 예측 저장 버튼
        const saveBtn = document.getElementById('save-prediction');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.savePrediction());
        }
        
        // 새 예측 버튼
        const newBtn = document.getElementById('new-prediction');
        if (newBtn) {
            newBtn.addEventListener('click', () => this.newPrediction());
        }
    }
    
    // ===== 번호 입력 관리 =====
    updatePreferredNumbers() {
        const inputs = document.querySelectorAll('.number-input');
        this.preferredNumbers = [];
        
        inputs.forEach(input => {
            const value = parseInt(input.value);
            if (value && value >= 1 && value <= 45) {
                this.preferredNumbers.push(value);
            }
        });
        
        // 중복 제거
        this.preferredNumbers = [...new Set(this.preferredNumbers)];
    }
    
    checkDuplicateNumbers() {
        const inputs = document.querySelectorAll('.number-input');
        const values = [];
        
        inputs.forEach(input => {
            input.classList.remove('input-duplicate');
            const value = parseInt(input.value);
            if (value && value >= 1 && value <= 45) {
                if (values.includes(value)) {
                    // 중복 발견
                    const duplicateInputs = Array.from(inputs).filter(inp => parseInt(inp.value) === value);
                    duplicateInputs.forEach(inp => {
                        inp.classList.add('input-duplicate');
                        this.showInputError(inp, '중복된 번호입니다');
                    });
                } else {
                    values.push(value);
                }
            }
        });
    }
    
    showInputError(input, message) {
        // 기존 에러 메시지 제거
        const existingError = input.parentElement.querySelector('.input-error-message');
        if (existingError) existingError.remove();
        
        // 새 에러 메시지 생성
        const errorMsg = document.createElement('div');
        errorMsg.className = 'input-error-message';
        errorMsg.textContent = message;
        input.parentElement.appendChild(errorMsg);
        
        // 자동 제거
        setTimeout(() => {
            errorMsg.remove();
        }, 3000);
    }
    
    hideInputError(input) {
        const errorMsg = input.parentElement.querySelector('.input-error-message');
        if (errorMsg) errorMsg.remove();
    }
    
    handleNumberPaste(pasteData, startIndex) {
        // 숫자만 추출
        const numbers = pasteData.match(/\d+/g);
        if (!numbers) return;
        
        const inputs = document.querySelectorAll('.number-input');
        let inputIndex = startIndex;
        
        numbers.forEach(numStr => {
            const num = parseInt(numStr);
            if (num >= 1 && num <= 45 && inputIndex < inputs.length) {
                inputs[inputIndex].value = num;
                inputIndex++;
            }
        });
        
        this.updatePreferredNumbers();
        this.checkDuplicateNumbers();
    }
    
    clearNumbers() {
        const inputs = document.querySelectorAll('.number-input');
        inputs.forEach(input => {
            input.value = '';
            input.classList.remove('input-error', 'input-duplicate');
        });
        
        // 에러 메시지 제거
        document.querySelectorAll('.input-error-message').forEach(msg => msg.remove());
        
        this.preferredNumbers = [];
        this.hideResult();
        
        window.showToast('번호가 초기화되었습니다', 'info', 2000);
    }
    
    fillRandomNumbers() {
        const inputs = document.querySelectorAll('.number-input');
        const randomNumbers = this.generateRandomLottoNumbers(6);
        
        inputs.forEach((input, index) => {
            if (index < randomNumbers.length) {
                input.value = randomNumbers[index];
                
                // 애니메이션 효과
                input.style.transform = 'scale(1.1)';
                input.style.background = 'rgba(102, 126, 234, 0.2)';
                
                setTimeout(() => {
                    input.style.transform = '';
                    input.style.background = '';
                }, 300);
            }
        });
        
        this.updatePreferredNumbers();
        window.showToast('랜덤 번호가 입력되었습니다', 'success', 2000);
    }
    
    // ===== 예측 생성 =====
    async generatePrediction() {
        try {
            // 입력 검증
            const validation = this.validateInput();
            if (!validation.valid) {
                window.showToast(validation.message, 'warning');
                return;
            }
            
            window.showLoading('AI가 최적의 번호를 생성중입니다...');
            
            // 선호 번호가 있으면 이를 기반으로, 없으면 완전 랜덤
            let predictedNumbers;
            
            if (this.preferredNumbers.length > 0) {
                predictedNumbers = await this.generateWithPreferences();
            } else {
                predictedNumbers = await this.generateRandomPrediction();
            }
            
            this.lastPrediction = {
                numbers: predictedNumbers,
                timestamp: new Date().toISOString(),
                method: this.preferredNumbers.length > 0 ? 'preferences' : 'random',
                preferences: [...this.preferredNumbers]
            };
            
            this.showPredictionResult(predictedNumbers);
            this.addToHistory(this.lastPrediction);
            
            window.showToast('AI 예측 완료!', 'success');
            
        } catch (error) {
            console.error('예측 생성 실패:', error);
            window.showToast('예측 생성 중 오류가 발생했습니다', 'error');
        } finally {
            window.hideLoading();
        }
    }
    
    async generateWithPreferences() {
        // 선호 번호를 포함한 최적의 조합 생성
        const needed = 6 - this.preferredNumbers.length;
        
        if (needed <= 0) {
            return this.preferredNumbers.slice(0, 6).sort((a, b) => a - b);
        }
        
        // AI 알고리즘 사용 (가장 정확도가 높은 것)
        if (window.githubManager && window.githubManager.algorithms) {
            try {
                const bestAlgorithm = this.findBestAlgorithm();
                const result = await window.githubManager.executeAlgorithm(bestAlgorithm);
                
                if (result && result.numbers) {
                    // 선호 번호와 AI 결과를 조합
                    return this.combinePreferencesWithAI(result.numbers);
                }
            } catch (error) {
                console.log('AI 알고리즘 사용 실패, 통계 기반 생성 사용');
            }
        }
        
        // 폴백: 통계 기반 생성
        return this.generateStatisticalPrediction();
    }
    
    async generateRandomPrediction() {
        // 가장 좋은 알고리즘 사용
        if (window.githubManager && window.githubManager.algorithms) {
            try {
                const bestAlgorithm = this.findBestAlgorithm();
                const result = await window.githubManager.executeAlgorithm(bestAlgorithm);
                
                if (result && result.numbers) {
                    return result.numbers;
                }
            } catch (error) {
                console.log('AI 알고리즘 사용 실패');
            }
        }
        
        // 폴백: 랜덤 생성
        return this.generateRandomLottoNumbers(6);
    }
    
    findBestAlgorithm() {
        const algorithms = window.githubManager.algorithms;
        let bestAlgorithm = 'super_v1';
        let bestAccuracy = 0;
        
        Object.entries(algorithms).forEach(([id, algo]) => {
            const accuracy = parseFloat(algo.accuracy) || 0;
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestAlgorithm = id;
            }
        });
        
        return bestAlgorithm;
    }
    
    combinePreferencesWithAI(aiNumbers) {
        const combined = [...this.preferredNumbers];
        const needed = 6 - combined.length;
        
        // AI 결과에서 선호번호와 중복되지 않는 번호 선택
        const available = aiNumbers.filter(num => !combined.includes(num));
        
        // 필요한 만큼 추가
        for (let i = 0; i < needed && i < available.length; i++) {
            combined.push(available[i]);
        }
        
        // 부족한 경우 랜덤으로 채우기
        while (combined.length < 6) {
            const randomNum = Math.floor(Math.random() * 45) + 1;
            if (!combined.includes(randomNum)) {
                combined.push(randomNum);
            }
        }
        
        return combined.sort((a, b) => a - b);
    }
    
    generateStatisticalPrediction() {
        // 간단한 통계 기반 예측 (실제로는 더 복잡한 로직)
        const hotNumbers = [7, 23, 31, 38, 42, 45]; // 예시 핫 넘버
        const combined = [...this.preferredNumbers];
        
        const needed = 6 - combined.length;
        const available = hotNumbers.filter(num => !combined.includes(num));
        
        // 핫 넘버에서 추가
        for (let i = 0; i < needed && i < available.length; i++) {
            combined.push(available[i]);
        }
        
        // 부족한 경우 랜덤 추가
        while (combined.length < 6) {
            const randomNum = Math.floor(Math.random() * 45) + 1;
            if (!combined.includes(randomNum)) {
                combined.push(randomNum);
            }
        }
        
        return combined.sort((a, b) => a - b);
    }
    
    generateRandomLottoNumbers(count) {
        const numbers = [];
        while (numbers.length < count) {
            const num = Math.floor(Math.random() * 45) + 1;
            if (!numbers.includes(num)) {
                numbers.push(num);
            }
        }
        return numbers.sort((a, b) => a - b);
    }
    
    // ===== 빠른 예측 =====
    async quickPrediction() {
        try {
            window.showLoading('빠른 AI 예측 생성중...');
            
            // 가장 빠른 알고리즘 사용
            const quickAlgorithm = this.findQuickestAlgorithm();
            let predictedNumbers;
            
            if (window.githubManager) {
                try {
                    const result = await window.githubManager.executeAlgorithm(quickAlgorithm);
                    predictedNumbers = result.numbers || this.generateRandomLottoNumbers(6);
                } catch (error) {
                    predictedNumbers = this.generateRandomLottoNumbers(6);
                }
            } else {
                predictedNumbers = this.generateRandomLottoNumbers(6);
            }
            
            this.lastPrediction = {
                numbers: predictedNumbers,
                timestamp: new Date().toISOString(),
                method: 'quick',
                algorithm: quickAlgorithm
            };
            
            this.showPredictionResult(predictedNumbers);
            this.addToHistory(this.lastPrediction);
            
            window.showToast('빠른 예측 완료!', 'success');
            
        } catch (error) {
            console.error('빠른 예측 실패:', error);
            window.showToast('빠른 예측 중 오류가 발생했습니다', 'error');
        } finally {
            window.hideLoading();
        }
    }
    
    findQuickestAlgorithm() {
        if (!window.githubManager || !window.githubManager.algorithms) {
            return 'ultimate_v1';
        }
        
        const algorithms = window.githubManager.algorithms;
        let quickestAlgorithm = 'ultimate_v1';
        let quickestTime = 10; // 기본값
        
        Object.entries(algorithms).forEach(([id, algo]) => {
            const timeStr = algo.execution_time || '~2초';
            const time = parseFloat(timeStr.replace(/[^\d.]/g, ''));
            
            if (time < quickestTime) {
                quickestTime = time;
                quickestAlgorithm = id;
            }
        });
        
        return quickestAlgorithm;
    }
    
    // ===== 결과 표시 =====
    showPredictionResult(numbers) {
        const resultContainer = document.getElementById('prediction-result');
        const numbersContainer = document.getElementById('recommended-numbers');
        
        if (numbersContainer) {
            numbersContainer.innerHTML = numbers.map((num, index) => `
                <div class="prediction-number" style="animation-delay: ${index * 0.1}s">
                    <span>${num}</span>
                </div>
            `).join('');
        }
        
        if (resultContainer) {
            resultContainer.classList.remove('hidden');
            
            // 스크롤 애니메이션
            resultContainer.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
            
            // 강조 효과
            setTimeout(() => {
                resultContainer.style.transform = 'scale(1.02)';
                resultContainer.style.boxShadow = '0 0 30px rgba(102, 126, 234, 0.3)';
                
                setTimeout(() => {
                    resultContainer.style.transform = '';
                    resultContainer.style.boxShadow = '';
                }, 1000);
            }, 500);
        }
    }
    
    hideResult() {
        const resultContainer = document.getElementById('prediction-result');
        if (resultContainer) {
            resultContainer.classList.add('hidden');
        }
    }
    
    // ===== 예측 저장 =====
    async savePrediction() {
        if (!this.lastPrediction) {
            window.showToast('저장할 예측이 없습니다', 'warning');
            return;
        }
        
        // ✅ 번호 정규화 - 배열로 확실히 변환
        const normalizedNumbers = this.normalizeNumbers(this.lastPrediction.numbers);
        
        console.log('💾 예측 저장 시도:', {
            original: this.lastPrediction.numbers,
            normalized: normalizedNumbers,
            algorithm: this.lastPrediction.algorithm || 'manual',
            method: this.lastPrediction.method
        });
        
        try {
            // ✅ 서버 전송 데이터 구성
            const payload = {
                numbers: normalizedNumbers, // 반드시 배열
                algorithm: this.lastPrediction.algorithm || 'manual',
                algorithm_name: this.lastPrediction.algorithm_name || 'AI 예측',
                timestamp: this.lastPrediction.timestamp,
                method: this.lastPrediction.method,
                preferences: this.lastPrediction.preferences || [],
                round_predicted: 1191
            };
            
            console.log('📤 서버 전송 데이터:', JSON.stringify(payload, null, 2));
            
            const response = await fetch('/api/save-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                console.log('✅ 예측 저장 완료:', result.prediction_id);
                window.showToast('예측이 저장되었습니다!', 'success');
                
                // 통계 업데이트
                if (window.lottoApp) {
                    window.lottoApp.loadUserData();
                }
                
                // 저장 버튼 상태 업데이트
                const saveBtn = document.getElementById('save-prediction');
                if (saveBtn) {
                    saveBtn.innerHTML = '<i class="fas fa-check mr-2"></i>저장됨';
                    saveBtn.disabled = true;
                    saveBtn.classList.add('btn-saved');
                }
                
            } else {
                throw new Error(result.message || '저장 실패');
            }
            
        } catch (error) {
            console.error('❌ 예측 저장 실패:', error);
            window.showToast('네트워크 오류가 발생했습니다', 'error');
        }
    }
    
    /**
     * ✅ 번호 정규화 - 다양한 입력을 배열로 변환
     */
    normalizeNumbers(numbers) {
        // 이미 배열이면 그대로
        if (Array.isArray(numbers)) {
            return numbers.map(n => parseInt(n)).filter(n => !isNaN(n) && n >= 1 && n <= 45);
        }
        
        // 문자열인 경우
        if (typeof numbers === 'string') {
            // 쉼표 구분
            if (numbers.includes(',')) {
                return numbers.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n) && n >= 1 && n <= 45);
            }
            // 공백 구분
            if (numbers.includes(' ')) {
                return numbers.split(/\s+/).map(n => parseInt(n.trim())).filter(n => !isNaN(n) && n >= 1 && n <= 45);
            }
        }
        
        // 숫자인 경우
        if (typeof numbers === 'number') {
            if (numbers >= 1 && numbers <= 45) {
                return [numbers];
            }
        }
        
        console.warn('⚠️ 번호 정규화 실패, 빈 배열 반환:', numbers);
        return [];
    }
    
    // ===== 새 예측 =====
    newPrediction() {
        this.clearNumbers();
        this.lastPrediction = null;
        
        // 저장 버튼 초기화
        const saveBtn = document.getElementById('save-prediction');
        if (saveBtn) {
            saveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>예측 저장';
            saveBtn.disabled = false;
            saveBtn.classList.remove('btn-saved');
        }
        
        window.showToast('새로운 예측을 시작합니다', 'info');
    }
    
    // ===== 검증 및 유틸리티 =====
    validateInput() {
        const inputs = document.querySelectorAll('.number-input');
        const duplicateInputs = document.querySelectorAll('.input-duplicate');
        const errorInputs = document.querySelectorAll('.input-error');
        
        if (duplicateInputs.length > 0) {
            return {
                valid: false,
                message: '중복된 번호가 있습니다'
            };
        }
        
        if (errorInputs.length > 0) {
            return {
                valid: false,
                message: '잘못된 번호가 있습니다 (1-45 범위)'
            };
        }
        
        return { valid: true };
    }
    
    // ===== 예측 기록 관리 =====
    addToHistory(prediction) {
        this.predictionHistory.unshift(prediction);
        
        // 최대 50개까지만 보관
        if (this.predictionHistory.length > 50) {
            this.predictionHistory.pop();
        }
        
        this.savePredictionHistory();
    }
    
    loadPredictionHistory() {
        try {
            const stored = localStorage.getItem('lotto_prediction_history');
            if (stored) {
                this.predictionHistory = JSON.parse(stored);
            }
        } catch (error) {
            console.error('예측 기록 로드 실패:', error);
            this.predictionHistory = [];
        }
    }
    
    savePredictionHistory() {
        try {
            localStorage.setItem('lotto_prediction_history', JSON.stringify(this.predictionHistory));
        } catch (error) {
            console.error('예측 기록 저장 실패:', error);
        }
    }
    
    getPredictionStats() {
        return {
            totalPredictions: this.predictionHistory.length,
            methodStats: this.getMethodStats(),
            recentPredictions: this.predictionHistory.slice(0, 10),
            favoriteNumbers: this.getFavoriteNumbers()
        };
    }
    
    getMethodStats() {
        const stats = {};
        this.predictionHistory.forEach(pred => {
            const method = pred.method || 'unknown';
            stats[method] = (stats[method] || 0) + 1;
        });
        return stats;
    }
    
    getFavoriteNumbers() {
        const numberCounts = {};
        
        this.predictionHistory.forEach(pred => {
            pred.numbers.forEach(num => {
                numberCounts[num] = (numberCounts[num] || 0) + 1;
            });
        });
        
        return Object.entries(numberCounts)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10)
            .map(([num, count]) => ({ number: parseInt(num), count }));
    }
}

// ===== 전역 인스턴스 및 함수 =====
let predictionManager;

function initQuickPrediction() {
    if (!predictionManager) {
        predictionManager = new PredictionManager();
    }
}

function updateStats() {
    // 통계 업데이트 (별도 구현)
    console.log('통계 업데이트 시뮬레이션');
}

// ===== 초기화 =====
document.addEventListener('DOMContentLoaded', function() {
    predictionManager = new PredictionManager();
    
    // 전역 접근을 위해 window에 할당
    window.predictionManager = predictionManager;
    window.initQuickPrediction = initQuickPrediction;
    window.updateStats = updateStats;
});
