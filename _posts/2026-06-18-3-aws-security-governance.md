---
title: "[AWS 자격증 #3] 보안 서비스와 거버넌스"
date: 2026-06-18 10:20:00 +0900
slug: aws-security-governance
categories:
  - AWS
tags:
  - AWS
  - CCP
  - CLF-C02
  - 보안
  - 거버넌스
toc: true
toc_sticky: true
---

<!-- TODO: 이전 글(공유 책임 모델과 IAM)과 연결하는 도입부. 헤더 없이 본문 시작.
"누가 책임지는지"는 정했으니, 이제 "어떤 도구로 지키고 감시하는지"로 이어가는 흐름 -->

## 보안 거버넌스란

<!-- TODO: 단순히 "막는 것"과 "감시/증명하는 것"의 차이 설명 -->

## 핵심 개념과 구조

<!-- TODO: 다이어그램 1개 — 탐지(GuardDuty/Inspector) → 통합관제(Security Hub) → 대응 흐름 -->

다룰 내용:
- 위협 탐지: Amazon GuardDuty (이상행동 탐지), Amazon Inspector (취약점 스캔), Amazon Detective, Amazon Macie(민감정보 탐지)
- 통합 관제: AWS Security Hub
- 네트워크 보호: AWS WAF, AWS Shield, AWS Firewall Manager
- 암호화: 전송 중(in transit) vs 저장 중(at rest) 암호화, AWS KMS, AWS CloudHSM, ACM(인증서 관리)
- 모니터링/감사: Amazon CloudWatch(모니터링), AWS CloudTrail(API 호출 기록), AWS Config(리소스 설정 변경 추적), AWS Audit Manager(컴플라이언스 증빙)
- 컴플라이언스 자료: AWS Artifact (감사 보고서/약관 다운로드)
- 운영 최적화 점검: AWS Trusted Advisor
- 서드파티 보안 제품: AWS Marketplace
- 지원 채널: AWS Knowledge Center, AWS Security Blog

## 어디에 쓰이는가

<!-- TODO: 실무에서 보안팀이 GuardDuty/Security Hub 알림 받고 대응하는 흐름 예시 -->

## 직접 해보기 (AWS 콘솔)

<!-- TODO: GuardDuty 활성화해보기, CloudTrail 이벤트 히스토리 조회해보기 안내 -->

## 시험 포인트

<!-- TODO: 헷갈리기 쉬운 비교
- GuardDuty(위협 탐지) vs Inspector(취약점 스캔) vs Macie(민감정보) 역할 구분
- CloudTrail(누가 무엇을 했는지) vs CloudWatch(성능/상태 모니터링) vs Config(설정 변경 이력) 구분
- WAF(애플리케이션 계층 공격 방어) vs Shield(DDoS 방어) 구분 -->

## 정리

<!-- TODO: 표 정리 -->

다음 글: **[컴퓨팅 서비스](/aws/aws-compute-services/)**
