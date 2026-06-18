---
title: "[AWS 자격증 #4] 컴퓨팅 서비스"
date: 2026-06-18 10:30:00 +0900
slug: aws-compute-services
categories:
  - AWS
tags:
  - AWS
  - CCP
  - CLF-C02
  - EC2
  - 컴퓨팅
toc: true
toc_sticky: true
---

<!-- TODO: 이전 글(보안)과 연결하는 도입부. 헤더 없이 본문 시작.
보안은 다 둘러봤으니, 이제 실제로 "AWS에서 코드를 어떻게 돌리는지"로 전환 -->

## 컴퓨팅 서비스란

<!-- TODO: 온프렘 서버 vs AWS 컴퓨팅 옵션들의 스펙트럼(직접 관리 ↔ 완전 관리형) 설명 -->

## 핵심 개념과 구조

<!-- TODO: 다이어그램 1개 — EC2(서버) → 컨테이너(ECS/EKS) → 서버리스(Lambda/Fargate) 관리 부담 스펙트럼 -->

다룰 내용:
- 배포/운영 방식: 콘솔 vs API/SDK/CLI vs IaC(CloudFormation), 클라우드/하이브리드/온프렘 모델
- Amazon EC2: 인스턴스 유형, On-Demand/Reserved/Spot 등 구매 옵션 (요금제는 9번 글에서 깊게 다룸, 여기선 개념만)
- 컨테이너: Amazon ECS, Amazon EKS, Amazon ECR
- 서버리스: AWS Fargate, AWS Lambda
- Auto Scaling — 트래픽에 따라 자동으로 늘리고 줄이기
- 로드 밸런서(ELB) — 트래픽 분산
- 기타: AWS Batch, AWS Elastic Beanstalk, Amazon Lightsail, AWS Outposts

## 어디에 쓰이는가

<!-- TODO: 트래픽 패턴별로 EC2 vs Lambda vs ECS/EKS 중 무엇을 선택하는지 실무 기준 -->

## 직접 해보기 (AWS 콘솔)

<!-- TODO: EC2 프리티어 인스턴스 띄워보기, Lambda 함수 하나 만들어서 실행해보기 안내 -->

## 시험 포인트

<!-- TODO: 헷갈리기 쉬운 비교
- ECS vs EKS (AWS 자체 오케스트레이션 vs 쿠버네티스)
- Fargate vs Lambda (컨테이너 서버리스 vs 함수 서버리스)
- On-Demand vs Reserved vs Spot 인스턴스 — 언제 무엇을 쓰는지 -->

## 정리

<!-- TODO: 표 정리 -->

다음 글: **[스토리지 서비스](/aws/aws-storage-services/)**
