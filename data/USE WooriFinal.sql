USE WooriFinal


CREATE TABLE plan_input (
    id INT AUTO_INCREMENT PRIMARY KEY,                      -- 계획 고유 ID (기본 키)
    user_id INT NOT NULL,                             -- 이 계획의 소유자 (FK, 새로 추가됨)
    target_house_price BIGINT,                        -- 목표 주택 가격 (타입 변경)
    target_location VARCHAR(100),                           -- 주택 위치
    housing_type VARCHAR(50),                               -- 주거지 형태
    available_assets BIGINT,                          -- 현재 사용 가능한 자산 (타입 변경)
    income_usage_ratio INT,                           -- 소득 활용 비율 (%) (타입 변경)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 계획 생성일 (추가 권장)
    -- 외래 키(FK) 제약 조건 설정
    -- 이 테이블의 user_id는 반드시 user_info(user_id)에 존재해야 함
    CONSTRAINT fk_plan_user
        FOREIGN KEY (user_id) 
        REFERENCES user_info(user_id)
        ON DELETE CASCADE  -- 사용자가 탈퇴하면(user_info에서 삭제되면) 계획도 함께 삭제
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DROP Table plan_input;

CREATE TABLE state (
    id INT AUTO_INCREMENT PRIMARY KEY,
    region_cc VARCHAR(10) NOT NULL,
    region_nm VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    apartment_price BIGINT DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- -- 1️⃣ NULL 값 행 삭제
-- DELETE FROM state
-- WHERE apartment_price IS NULL;

-- -- 2️⃣ AUTO_INCREMENT 재정렬
-- ALTER TABLE state AUTO_INCREMENT = 1;

-- -- 3️⃣ id 값 자체를 다시 1부터 연속적으로 재배열 (선택사항)
-- SET @count = 0;
-- UPDATE state SET id = (@count := @count + 1) ORDER BY id;

-- ALTER TABLE state MODIFY COLUMN apartment_price BIGINT; -- apartment_price를 int -> BigInt로 변경


UPDATE state
SET apartment_price = apartment_price * 10000;
COMMIT;
-- 원단위로 변환

ALTER TABLE state ADD COLUMN multi_price BIGINT DEFAULT NULL; -- multi_price컬럼추가 (연립다세대)

ALTER TABLE state 
CHANGE COLUMN price apartment_price INT DEFAULT NULL;

ALTER TABLE state ADD COLUMN multi_price BIGINT DEFAULT NULL;

UPDATE state
SET multi_price = multi_price * 10000;
COMMIT;

UPDATE state
SET multi_price = multi_price / 10000;
COMMIT;

UPDATE state
SET apartment_price = apartment_price * 10000,
    multi_price = multi_price * 10000
WHERE region_cc = '41000';

-- DELETE FROM state
-- WHERE id BETWEEN 26 AND 282;




SET @COUNT = 0;

UPDATE state SET id = (@COUNT := @COUNT + 1)
ORDER BY id;

ALTER TABLE state MODIFY COLUMN id INT;
ALTER TABLE state DROP PRIMARY KEY;
ALTER TABLE state ADD PRIMARY KEY (region_cc);

ALTER TABLE state DROP COLUMN id;

CREATE TABLE loan_product (
    loan_id INT PRIMARY KEY AUTO_INCREMENT,
    loan_name VARCHAR(100),
    loan_type VARCHAR(50),
    interest_type VARCHAR(50),
    interest_rate DECIMAL(5,2),
    max_ltv INT,
    max_dsr INT,
    repayment_method VARCHAR(50),
    period_years INT,
    description TEXT
);

INSERT INTO loan_product (loan_name, loan_type, interest_type, interest_rate, max_ltv, max_dsr, repayment_method, period_years, description)
VALUES
('우리 아파트론', '주택담보대출', '고정금리', 3.85, 70, 40, '원리금균등상환', 30, '아파트를 담보로 최대 70%까지 대출 가능'),
('우리 직장인 신용대출', '신용대출', '변동금리', 4.65, NULL, 40, '만기일시상환', 5, '급여소득자 대상 신용대출 상품'),
('우리 전세자금대출', '전세자금대출', '고정금리', 3.25, NULL, 40, '원금균등상환', 2, '임차보증금의 최대 80%까지 지원'),
('우리 청년 희망대출', '청년우대대출', '고정금리', 2.75, 80, 40, '원리금균등상환', 20, '만 34세 이하 청년 대상 금리 우대 상품'),
('우리 신혼부부 주택대출', '신혼우대대출', '변동금리', 3.15, 80, 40, '원금균등상환', 25, '신혼부부 전용 주택구입자금 상품'),
('우리 중도금대출', '주택담보대출', '변동금리', 3.95, 60, 40, '만기일시상환', 3, '분양주택 계약자의 중도금 납부용 상품'),
('우리 스마트 모기지', '주택담보대출', '고정금리', 4.05, 70, 40, '원리금균등상환', 35, '비대면으로 가능한 아파트 담보대출'),
('우리 서민 안심대출', '서민대출', '고정금리', 2.85, 70, 40, '원리금균등상환', 20, '소득 5천만 원 이하 서민 대상'),
('우리 마이너스통장대출', '신용대출', '변동금리', 5.20, NULL, 40, '한도대출', 1, '마이너스통장 형태의 신용대출'),
('우리 전세플러스대출', '전세자금대출', '변동금리', 3.55, NULL, 40, '원리금균등상환', 3, '보증금 5억 이하 전세자 대상');

DROP TABLE loan_product

CREATE TABLE user_info (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    gender ENUM('M','F') NOT NULL,
    region VARCHAR(100),
    income BIGINT,
    monthly_salary BIGINT,
    job_type VARCHAR(50),
    employment_years INT,
    credit_score INT,
    existing_loans INT,
    total_debt BIGINT,
    savings_balance BIGINT,
    investment_balance BIGINT,
    operating_income BIGINT,
    annual_revenue BIGINT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    -- 🔽 최근 대출 추천 결과 저장용
    last_recommended_loan_id INT NULL,
    last_loan_amount BIGINT NULL,
    last_monthly_payment BIGINT NULL,
    last_shortage_amount BIGINT NULL,
    last_recommend_date DATETIME DEFAULT NULL
);


INSERT INTO user_info
(name, age, gender, region, income, monthly_salary, job_type, employment_years, credit_score, existing_loans, total_debt, savings_balance, investment_balance, operating_income, annual_revenue)
VALUES
-- ① 직장인 (김도현)
('김도현', 31, 'M', '서울특별시 송파구', 55000000, 4600000, '직장인', 4, 820, 1, 35000000, 20000000, 15000000, NULL, NULL),

-- ② 공무원 (이서연)
('이서연', 34, 'F', '부산광역시 해운대구', 64000000, 5300000, '공무원', 7, 840, 0, 15000000, 35000000, 10000000, NULL, NULL),

-- ③ 자영업자 (박민수)
('박민수', 40, 'M', '대구광역시 수성구', 72000000, NULL, '자영업', 10, 780, 2, 80000000, 25000000, 20000000, 36000000, 180000000),

-- ④ 프리랜서 (정하은)
('정하은', 29, 'F', '인천광역시 연수구', 48000000, NULL, '프리랜서', 3, 760, 1, 20000000, 12000000, 8000000, 15000000, 90000000),

-- ⑤ 신혼부부 (최준혁)
('최준혁', 33, 'M', '경기도 수원시 영통구', 58000000, 4800000, '직장인', 5, 810, 0, 10000000, 30000000, 25000000, NULL, NULL);


ALTER TABLE plan_input
ADD COLUMN loan_amount BIGINT NULL AFTER target_period_years,
ADD COLUMN recommended_loan_id INT NULL AFTER loan_amount;

ALTER TABLE user_info
ADD COLUMN last_recommended_loan_id INT NULL AFTER investment_balance,
ADD COLUMN last_loan_amount BIGINT NULL AFTER last_recommended_loan_id,
ADD COLUMN last_monthly_payment BIGINT NULL AFTER last_loan_amount,
ADD COLUMN last_shortage_amount BIGINT NULL AFTER last_monthly_payment,
ADD COLUMN last_recommend_date DATETIME DEFAULT CURRENT_TIMESTAMP AFTER last_shortage_amount;

INSERT INTO loan_product
(loan_id, loan_name, loan_type, interest_type, interest_rate, max_ltv, max_dsr, repayment_method, period_years, description)
VALUES
(1, '우리 청년 희망대출', '신용대출', '고정금리', 2.75, 80, 40, '원리금균등상환', 20, '만 34세 이하 청년 대상 금리 우대 상품'),
(2, '우리 직장인 대출', '신용대출', '변동금리', 3.25, 70, 40, '원리금균등상환', 15, '근속 1년 이상 직장인 대상 상품'),
(3, '우리 프리랜서 안심대출', '신용대출', '고정금리', 3.80, 60, 35, '원리금균등상환', 10, '프리랜서 및 사업소득자 대상 상품'),
(4, '우리 주택담보대출', '담보대출', '변동금리', 3.10, 90, 45, '원리금균등상환', 30, '아파트, 오피스텔 담보 가능 상품'),
(5, '우리 신혼부부 전용대출', '담보대출', '고정금리', 2.90, 80, 50, '원리금균등상환', 25, '신혼부부 및 무주택자 우대금리 제공')
ON DUPLICATE KEY UPDATE
loan_name = VALUES(loan_name),
loan_type = VALUES(loan_type),
interest_type = VALUES(interest_type),
interest_rate = VALUES(interest_rate),
max_ltv = VALUES(max_ltv),
max_dsr = VALUES(max_dsr),
repayment_method = VALUES(repayment_method),
period_years = VALUES(period_years),
description = VALUES(description);

ALTER TABLE plan_input
ADD COLUMN remaining_after_loan BIGINT NULL AFTER loan_amount;


-- 대출 상품 테이블 생성
CREATE TABLE loan_product (
    product_id INT AUTO_INCREMENT PRIMARY KEY,  -- 상품 식별을 위한 고유 ID (추가하는 것을 권장)
    
    product_name VARCHAR(255),
    bank_name VARCHAR(255),
    product_type VARCHAR(255),
    summary TEXT,
    features TEXT,
    target_customer TEXT,
    target_housing_type VARCHAR(255),
    limit_description TEXT,
    period_description TEXT,
    repayment_method VARCHAR(255),
    rate_type VARCHAR(255),
    rate_description TEXT,
    preferential_rate_info TEXT,
    prepayment_penalty_desc TEXT,
    collateral_description TEXT,
    application_method VARCHAR(255),
    application_period_desc TEXT,
    required_documents TEXT,
    customer_costs TEXT,
    late_fee_rate VARCHAR(255),
    interest_calculation VARCHAR(255),
    interest_payment_method TEXT
);

INSERT INTO loan_product (
    product_name,
    bank_name,
    product_type,
    summary,
    features,
    target_customer,
    target_housing_type,
    limit_description,
    period_description,
    repayment_method,
    rate_type,
    rate_description,
    preferential_rate_info,
    prepayment_penalty_desc,
    collateral_description,
    application_method,
    application_period_desc,
    required_documents,
    customer_costs,
    late_fee_rate,
    interest_calculation,
    interest_payment_method
) VALUES (
    '스마트징검다리론',
    '우리은행',
    '중도금대출',
    '쉽고 간편한 모바일 중도금대출',
    '인터넷/스마트뱅킹 전용 중도금대출',
    '분양 계약금을 지급한 분양계약자',
    '공동주택, 주상복합',
    '분양가 또는 조합원부담금의 60% 이내 (최저/최고 한도 없음)',
    '시행사/시공사와 은행이 협약한 기한 내 (만기 시 심사 후 연장 가능)',
    '만기일시상환',
    '선택형 (고정금리 또는 변동금리)',
    '기준금리 + 가산금리 (시행/시공사와 은행이 협약한 금리)',
    '집단대출 금리 협약에 따라 개별 차주 금리 동일',
    '시행주체와 은행 협약에 따름',
    '신용, 시공사연대보증, 한국주택금융공사보증서, 주택도시보증공사보증서 등',
    '인터넷뱅킹, 모바일뱅킹',
    '시행사/시공사와 은행 협약에 따라 별도 통보',
    '연소득증빙서류(근로소득원천징수영수증, 소득금액증명원 등), 주민등록등본(또는 국내거소신고서류), 분양계약서, 건강보험자격득실확인서',
    '인지세 (대출금액별 차등, 은행/고객 50% 부담), 보증료 (고객 부담)',
    '대출금리 + 연 3% (최고 연 12%)',
    '대출금액 * 대출이자율 * 이자일수 / 365(윤년 366)',
    '매월 후취 (자동이체)'
);

-- 대출금 컬럼 생성
ALTER TABLE user_info
ADD COLUMN loan_amount BIGINT;

ALTER TABLE state ADD COLUMN officetel_price BIGINT DEFAULT NULL;
ALTER TABLE state ADD COLUMN detached_price BIGINT DEFAULT NULL;