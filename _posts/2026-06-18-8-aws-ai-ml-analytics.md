---
title: "[AWS 자격증 #8] AI/ML과 분석, 그리고 기타 서비스"
date: 2026-06-18 11:10:00 +0900
slug: aws-ai-ml-analytics
categories:
  - AWS
tags:
  - AWS
  - CCP
  - CLF-C02
  - AI
  - 머신러닝
  - 분석
toc: true
toc_sticky: true
---

<!-- TODO: 이전 글(데이터베이스)과 연결하는 도입부. 헤더 없이 본문 시작.
저장된 데이터를 가지고 "분석하고 예측하는" 서비스들로 전환.
※ 이 포스트는 공식 exam guide의 Task 3.7(AI/ML·분석 서비스)을 다루는데,
원래 9개 구성안에는 빠져있던 부분이라 그래프/검증 보강에 좀 더 신경 쓸 것 -->

## AI/ML과 분석 서비스란

<!-- TODO: "직접 모델을 만드는 것" vs "이미 만들어진 AI 기능을 API로 가져다 쓰는 것" 구분 설명 -->

## 핵심 개념과 구조

<!-- TODO: 다이어그램 1개 — 데이터 수집(Kinesis) → 가공(Glue) → 저장/분석(Athena/Redshift) → 시각화(QuickSight) 파이프라인 -->

다룰 내용:
- 머신러닝(완전관리형 AI 서비스, 직접 학습 불필요한 것 위주):
  - Amazon SageMaker AI — 직접 모델을 만들고 학습/배포하는 종합 플랫폼
  - Amazon Lex — 챗봇/대화형 인터페이스
  - Amazon Kendra — 지능형 검색
  - (간략) Comprehend, Polly, Rekognition, Textract, Transcribe, Translate, Amazon Q
- 분석 서비스:
  - Amazon Athena — S3 데이터를 SQL로 바로 조회
  - Amazon Kinesis — 실시간 스트리밍 데이터 처리
  - AWS Glue — ETL(데이터 추출/변환/적재)
  - Amazon QuickSight — BI 시각화 대시보드
  - (간략) Amazon EMR, Amazon OpenSearch Service, Amazon Redshift
- 기타 in-scope 카테고리 (Task 3.8, 간략 정리표로):
  - 앱 통합: Amazon EventBridge, SNS, SQS, Step Functions
  - 비즈니스 애플리케이션: Amazon Connect, Amazon SES
  - 개발자 도구: AWS CodeBuild, CodePipeline, X-Ray, AWS CLI
  - End User Computing: Amazon AppStream 2.0, WorkSpaces
  - 프론트엔드/모바일: AWS Amplify, AWS AppSync
  - IoT: AWS IoT Core

## 어디에 쓰이는가

<!-- TODO: 고객 문의 챗봇(Lex), 사내 문서 검색(Kendra), 매출 대시보드(QuickSight) 같은 실무 예시 -->

## 직접 해보기 (AWS 콘솔)

<!-- TODO: S3에 작은 CSV 올리고 Athena로 SQL 조회해보기 안내 -->

## 시험 포인트

<!-- TODO: 헷갈리기 쉬운 비교
- SageMaker AI(직접 모델 구축) vs Lex/Kendra/Rekognition 등(완성된 AI 기능 API)
- Athena(서버리스 즉석 조회) vs Redshift(전통적 데이터 웨어하우스) vs QuickSight(시각화 도구) 역할 구분
- SNS(발행-구독, 알림) vs SQS(메시지 큐, 순서/버퍼링) 구분 -->

## 정리

<!-- TODO: 표 정리 -->

다음 글: **[요금제와 비용 관리](/aws/aws-pricing-billing/)**
