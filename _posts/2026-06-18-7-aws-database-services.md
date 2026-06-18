---
title: "[AWS 자격증 #7] 데이터베이스 서비스"
date: 2026-06-18 11:00:00 +0900
slug: aws-database-services
categories:
  - AWS
tags:
  - AWS
  - CCP
  - CLF-C02
  - 데이터베이스
toc: true
toc_sticky: true
---

<!-- TODO: 이전 글(네트워킹)과 연결하는 도입부. 헤더 없이 본문 시작.
연결된 자원들 안에 결국 "데이터를 어디에 저장하는지"로 전환 -->

## 데이터베이스 서비스란

<!-- TODO: EC2에 DB를 직접 설치하는 것 vs 관리형 DB 서비스를 쓰는 것의 차이
("직접 차를 수리하는 것 vs 정비소에 맡기는 것" 비유 고려) -->

## 핵심 개념과 구조

<!-- TODO: 다이어그램 1개 — 관계형(RDS/Aurora) vs NoSQL(DynamoDB) vs 캐시(ElastiCache) 분류 -->

다룰 내용:
- EC2에 직접 설치한 DB vs 관리형 DB의 책임 범위 차이 (보안 글의 공유 책임 모델과 연결)
- 관계형 DB: Amazon RDS, Amazon Aurora
- NoSQL: Amazon DynamoDB
- 메모리 기반 캐시: Amazon ElastiCache
- 기타 관리형 DB: Amazon DocumentDB, Amazon Neptune (간략히)
- 마이그레이션: AWS Database Migration Service(DMS), AWS Schema Conversion Tool(SCT)

## 어디에 쓰이는가

<!-- TODO: 트랜잭션 많은 서비스(RDS) vs 대규모 키-값 조회(DynamoDB) vs 캐싱(ElastiCache) 선택 기준 -->

## 직접 해보기 (AWS 콘솔)

<!-- TODO: RDS 프리티어 인스턴스 만들어보기, DynamoDB 테이블 만들고 아이템 넣어보기 안내 -->

## 시험 포인트

<!-- TODO: 헷갈리기 쉬운 비교
- RDS vs Aurora (호환성 vs 성능/가용성)
- 관계형(RDS) vs NoSQL(DynamoDB) 언제 무엇을 쓰는지
- DMS(데이터 이전) vs SCT(스키마 변환) 역할 구분 -->

## 정리

<!-- TODO: 표 정리 -->

다음 글: **[AI/ML과 분석, 그리고 기타 서비스](/aws/aws-ai-ml-analytics/)**
