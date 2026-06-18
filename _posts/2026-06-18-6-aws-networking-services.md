---
title: "[AWS 자격증 #6] 네트워킹 서비스"
date: 2026-06-18 10:50:00 +0900
slug: aws-networking-services
categories:
  - AWS
tags:
  - AWS
  - CCP
  - CLF-C02
  - VPC
  - 네트워킹
toc: true
toc_sticky: true
---

<!-- TODO: 이전 글(스토리지)과 연결하는 도입부. 헤더 없이 본문 시작.
컴퓨팅·스토리지 자원들이 서로 "어떻게 연결되는지"로 전환 -->

## 네트워킹 서비스란

<!-- TODO: VPC를 "AWS 안에 만드는 나만의 사설 네트워크"로 비유 설명 -->

## 핵심 개념과 구조

<!-- TODO: 다이어그램 1개 — Region > AZ > VPC > Subnet 계층 구조 -->

다룰 내용:
- 글로벌 인프라: Region, Availability Zone(AZ), Edge Location — Multi-AZ로 고가용성 확보, Multi-Region을 쓰는 경우
- VPC 구성요소: 서브넷(퍼블릭/프라이빗), 인터넷 게이트웨이, NAT 게이트웨이
- 보안 경계: 네트워크 ACL vs 보안 그룹
- DNS: Amazon Route 53
- 온프렘 연결: AWS Site-to-Site VPN, AWS Client VPN, AWS Direct Connect
- 기타: Amazon CloudFront(CDN), Amazon API Gateway, AWS Global Accelerator, AWS PrivateLink, AWS Transit Gateway

## 어디에 쓰이는가

<!-- TODO: 회사 데이터센터와 AWS를 안전하게 연결하는 실무 시나리오 (VPN vs Direct Connect 선택 기준) -->

## 직접 해보기 (AWS 콘솔)

<!-- TODO: VPC 마법사로 퍼블릭/프라이빗 서브넷 만들어보기 안내 -->

## 시험 포인트

<!-- TODO: 헷갈리기 쉬운 비교
- 네트워크 ACL(서브넷 단위, 스테이트리스) vs 보안 그룹(인스턴스 단위, 스테이트풀)
- VPN(인터넷 통해 암호화 터널) vs Direct Connect(전용선) — 비용/지연/안정성 트레이드오프
- AZ(물리적으로 분리된 데이터센터) vs Region(지리적으로 분리된 묶음) 구분 -->

## 정리

<!-- TODO: 표 정리 -->

다음 글: **[데이터베이스 서비스](/aws/aws-database-services/)**
