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

데이터베이스 글까지는 데이터를 "어디에 저장하는가"를 다뤘다. 이번 글은 그 저장된 데이터를 가지고 무언가를 분석하고, 예측하고, 더 똑똑한 기능을 만드는 서비스들을 다룬다.
참고로 이 글은 시험 범위 Task 3.7(AI/ML·분석 서비스)에 해당하는데, 시리즈를 처음 기획할 때 9개 구성안에서 빠져있던 부분이라 이번에 보강했다.

## AI/ML과 분석 서비스란

AI/ML 서비스를 쓰는 방법은 크게 두 가지다. 하나는 **직접 모델을 만드는 것**이다. 데이터를 모으고, 알맞은 알고리즘을 골라 학습시키고, 성능을 튜닝하는 전체 과정을 내가 통제한다. 다른 하나는 **이미 만들어진 AI 기능을 API로 가져다 쓰는 것**이다. 이미지에서 텍스트를 추출하거나, 문장을 번역하거나, 챗봇을 만드는 것처럼 누군가 이미 잘 학습시켜놓은 모델을 호출만 해서 쓰는 방식이다. AWS는 두 방식을 모두 지원하는데, 일반적으로는 API로 가져다 쓰는 게 훨씬 빠르고 쉽다.

## 핵심 개념과 구조

### 데이터 파이프라인 — 수집 → 가공 → 저장/분석 → 시각화

![데이터 파이프라인 - 수집 → 가공 → 저장/분석 → 시각화](/assets/images/aws_data_pipeline.png)

분석 서비스들은 대체로 이 4단계 흐름 안에서 역할을 나눠 맡는다.

**수집 — Amazon Kinesis**: 실시간으로 쏟아지는 스트리밍 데이터(클릭 로그, 센서 데이터 등)를 받아낸다. 배치로 한 번에 모으는 게 아니라 지금 일어나는 일을 바로바로 받는다는 점이 특징이다.

**가공 — AWS Glue**: 여러 소스에서 모인 데이터를 정제하고 형식을 변환하는 **ETL(Extract-Transform-Load)** 서비스다. 서버를 직접 관리할 필요 없이 데이터 변환 작업을 서버리스로 실행한다.

**저장/분석 — Amazon Athena / Amazon Redshift**: 가공된 데이터를 분석하는 단계다.
- **Amazon Athena**: S3에 있는 데이터를 별도로 옮기지 않고 **SQL로 바로 조회**할 수 있는 서버리스 쿼리 서비스. 쿼리한 만큼만 비용을 낸다
- **Amazon Redshift**: 대규모 데이터를 정형화해 쌓아두고 복잡한 분석 쿼리를 빠르게 처리하는 전통적인 **데이터 웨어하우스**. 정기적으로 대량의 데이터를 반복 분석해야 할 때 적합하다
- 추가로 **Amazon EMR**(Hadoop/Spark 같은 빅데이터 프레임워크를 관리형으로 실행), **Amazon OpenSearch Service**(로그 분석, 검색에 특화)도 대규모 데이터 처리에 쓰인다

**시각화 — Amazon QuickSight**: 분석 결과를 대시보드와 그래프로 보여주는 **BI(비즈니스 인텔리전스)** 도구다. 비개발자도 클릭만으로 데이터를 탐색할 수 있게 해준다.

### 머신러닝 서비스

- **Amazon SageMaker AI**: 데이터 준비, 모델 학습, 튜닝, 배포까지 직접 모델을 만드는 전체 과정을 지원하는 종합 플랫폼. "직접 모델을 만드는" 쪽에 해당한다
- **Amazon Lex**: 음성/텍스트 기반 챗봇과 대화형 인터페이스를 만드는 서비스 (Alexa와 같은 기술 기반)
- **Amazon Kendra**: 사내 문서, 매뉴얼 등에서 자연어 질문으로 답을 찾아주는 지능형 검색 서비스
- 그 외 완성된 AI 기능을 API로 제공하는 서비스로 **Amazon Comprehend**(자연어 분석, 감정 분석), **Amazon Polly**(텍스트→음성), **Amazon Rekognition**(이미지/영상 분석), **Amazon Textract**(문서에서 텍스트 추출), **Amazon Transcribe**(음성→텍스트), **Amazon Translate**(번역), **Amazon Q**(생성형 AI 어시스턴트)가 있다. 이들은 모두 "이미 학습된 모델을 API로 호출"하는 방식이라는 공통점이 있다

### 기타 서비스 (Task 3.8 — 그 외 in-scope 카테고리)

시험에 나올 수 있는 나머지 서비스들을 카테고리별로 간단히 정리한다.

| 카테고리 | 서비스 | 한 줄 설명 |
|---|---|---|
| 앱 통합 | EventBridge / SNS / SQS / Step Functions | 이벤트 라우팅 / 발행-구독 알림 / 메시지 큐 / 워크플로 조율 |
| 비즈니스 애플리케이션 | Amazon Connect / Amazon SES | 클라우드 콜센터 / 이메일 발송 |
| 개발자 도구 | CodeBuild / CodePipeline / X-Ray / AWS CLI | 빌드 / CI-CD 파이프라인 / 분산 추적 디버깅 / 명령줄 도구 |
| End User Computing | AppStream 2.0 / WorkSpaces | 앱 스트리밍 / 클라우드 가상 데스크톱 |
| 프론트엔드·모바일 | AWS Amplify / AWS AppSync | 웹·모바일 앱 개발 플랫폼 / GraphQL API 관리 |
| IoT | AWS IoT Core | IoT 디바이스 연결·관리 |

이 표에 있는 서비스들은 깊게 다루기보다 "이름과 카테고리를 보고 무엇인지 떠올릴 수 있는" 수준으로만 알아두면 된다. 다만 **SNS vs SQS**는 자주 나오는 비교라 짚어둔다 — SNS는 발행-구독(publish-subscribe) 방식으로 여러 구독자에게 즉시 알림을 뿌리고, SQS는 메시지를 큐에 쌓아뒀다가 받는 쪽이 순서대로(또는 버퍼링하며) 처리하는 메시지 큐다.

## 어디에 쓰이는가

고객 문의를 처음 받는 자동 응답 챗봇은 **Lex**로 만들고, 사내 위키나 매뉴얼에서 직원이 자연어로 검색하게 하려면 **Kendra**를 쓴다. 영업팀이 매출 추이를 보는 대시보드는 Redshift나 Athena로 데이터를 모아 **QuickSight**로 시각화하는 식으로 구성한다. 직접 추천 모델을 만들어야 하는 정도로 요구사항이 구체적이라면 **SageMaker AI**로 처음부터 학습시킨다.

## 직접 해보기 (AWS 콘솔)

S3 버킷에 간단한 CSV 파일을 하나 올려보고, Athena 콘솔에서 그 파일 위치를 가리키는 테이블을 정의한 뒤 `SELECT * FROM 테이블명 LIMIT 10` 같은 SQL을 실행해보자. 별도 데이터베이스 서버 없이 S3에 있는 파일을 SQL로 바로 조회할 수 있다는 게 Athena의 핵심이다.

## 시험 포인트

- **SageMaker AI vs Lex/Kendra/Rekognition 등**: SageMaker AI는 직접 모델을 구축·학습하는 플랫폼, Lex/Kendra/Rekognition 등은 이미 완성된 AI 기능을 API로 호출하는 서비스
- **Athena vs Redshift vs QuickSight**: Athena는 S3 데이터를 서버리스로 즉석 조회, Redshift는 대량 데이터를 쌓아두고 분석하는 전통적 데이터 웨어하우스, QuickSight는 분석 결과를 시각화하는 BI 도구. 셋은 파이프라인에서 서로 다른 단계를 맡는다
- **SNS vs SQS**: SNS는 발행-구독 방식으로 여러 구독자에게 즉시 알림(순서 보장 X), SQS는 메시지를 큐에 쌓아 순서대로/버퍼링하며 처리(주로 1:1 처리)

## 정리

| 영역 | 서비스 | 역할 |
|---|---|---|
| 데이터 파이프라인 | Kinesis → Glue → Athena/Redshift → QuickSight | 수집 → 가공 → 저장/분석 → 시각화 |
| 머신러닝(직접 구축) | Amazon SageMaker AI | 모델 학습·튜닝·배포 전체 과정 |
| 머신러닝(완성형 API) | Lex / Kendra / Comprehend / Polly / Rekognition / Textract / Transcribe / Translate / Q | 챗봇 / 검색 / 자연어분석 / TTS / 이미지분석 / 문서추출 / STT / 번역 / 생성형 AI |
| 분석(추가) | EMR / OpenSearch | 빅데이터 프레임워크 / 로그·검색 특화 |
| 기타 in-scope | EventBridge·SNS·SQS·Step Functions 등 | 앱 통합·콜센터·개발자 도구·EUC·프론트엔드·IoT |

다음 글: **[요금제와 비용 관리](/aws/aws-pricing-billing/)**
