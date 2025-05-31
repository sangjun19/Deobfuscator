# 프로젝트 주제
- 바이너리 프로그램에서 제어 구조를 식별하는 도구 개발

## 주요내용
- 바이너리 코드에서 loo-switch 제어구조를 식별하는 도구 개발. 다양한 컴파일러 및 컴파일 옵션으로 컴파일되고, 난독화를 통해 변환된 바이너리코드를 고려함

## 필요지식/경험
- LLM : Fine tuning, few shot prompting
- Obfuscation : Opaque predicate, Flattening, Virtualize
- Tools
    - Obfuscator : Tigress, VMProtect, O-LLVM
    - Analysis : gcc, clang, IDA, LLVM Pass
- Language : C, Python, Assmebly, LLVM

## Paper
- LLM
    - Can  LLMsObfuscate Code?  A  Systematic Analysis  of  Large Language  Models  into Assembly  Code Obfuscation (Seyedreza  Mohseni, Seyedali  Mohammadi, Deepa  Tilwani,  Yash Saxena,  Gerald  Ketu Ndawula, Sriram Vema, Edward  Raff,  Manas Gaur), AAAI 2025
    - PalmTree: Learning an Assembly  Language Model for Instruction Embedding  (Xuezixiang Li, Yu Qu, Heng Yin), CCS 2021
- Obfuscation
    - Deobfuscating virtualized malware using Hex-Rays Decompiler(Georgy Kucherin), VB London 2023
    - Loki : Hardening Code Obfuscation Against Automated Attacks(Moritz Schloegel, Tim Blazytko, Moritz Contag, Cornelius Aschermann, and Julius Basler, Ruhr-Universität Bochum), USENIX 2022
    - VMProtect의 역공학 방해 기능 분석 및 Pin을 이용한 우회 방안(박성우, 박용수), KTCCS 2021
    - VMProtect 동작원리 분석 및 자동 역난독화 구현(방철호, 석재현, 이상진), JKIISC 2020 



## 참고
- 궁극적으로 가상화 난독화된 코드 역난독화 연구를 지원

## 참여 인원
- 컴퓨터융합학부 202002514 안상준
- 인공지능학과 202202487 박혜연
- 컴퓨터융합학부 202202602 손예진

## Blog
- 안상준 : https://velog.io/@sangjun19
- 박혜연 : https://m.blog.naver.com/p-yeye
- 손예진 : https://snejs.tistory.com/ 

- - -
<br></br>
# 주차별 활동
## 12주차
- 활동 개요 : 최종 발표
- 발표 영상 링크 : [12주차 발표 영상](https://youtu.be/VVOBg1LKbHc)
- 제출물 : 테스트 결과 문서, 최종발표 자료, 최종발표 영상
- PR : [12주차 issue](https://github.com/sangjun19/Deobfuscator/issues/90)

## 11주차
- 활동 개요 : 테스트 계획서
- 발표 영상 링크 : [11주차 발표 영상](https://youtu.be/FyQronpNaEg)
- 제출물 : 테스트 계획서 문서, 발표 자료, 발표 영상
- PR : [11주차 issue](https://github.com/sangjun19/Deobfuscator/issues/66)

## 7주차
- 활동 개요 : 시퀀스 다이어그램
- 발표 영상 링크 : [7주차 발표 영상](https://youtu.be/ieKRZBKuw2U)
- 제출물 : 시퀀스 다이어그램 문서, 발표 자료, 발표 영상
- PR : [7주차 issue](https://github.com/sangjun19/Deobfuscator/issues/77)

## 5주차
- 활동 개요 : 유스케이스 명세서
- 발표 영상 링크 : [5주차 발표 영상](https://youtu.be/ZXJIGnuEcfQ)
- 제출물 : 유스케이스 문서, 발표 자료, 발표 영상
- PR : [5주차 issue](https://github.com/sangjun19/Deobfuscator/issues/52)

## 4주차
- 활동 개요 : 문제 정의서
- 발표 영상 링크 : [4주차 발표 영상](https://youtu.be/H-mpD-Et9gI)
- 제출물 : 문제 정의서 문서, 발표 자료, 발표 영상
- PR : [4주차 issue](https://github.com/sangjun19/Deobfuscator/issues/31)

## 3주차
- 활동 개요 : 브레인스토밍
- 발표 영상 링크 : [3주차 발표 영상](https://youtu.be/SZjYWCbPGhQ)
- 제출물 : 브레인스토밍 결과 보고서, 발표 자료, 발표 영상
- PR : [3주차 issue](https://github.com/sangjun19/Deobfuscator/issues/20)

## 2주차
- 활동 개요 : 기존 논문 분석 및 문제점 개요서 작성
- 발표 영상 링크 : [2주차 발표 영상](https://youtu.be/Lb9hr2o6Qb4)
- 제출물 : 문제점 개요서, 발표 자료, 발표 영상
- PR : [2주차 issue](https://github.com/sangjun19/Deobfuscator/issues/13)

## 1주차
- 활동 개요 : 연구 개요서 작성
- 발표 영상 링크 : [1주차 발표 영상](https://youtu.be/Vtu4uO13c0s)
- 제출물 : 연구 개요서, 발표 자료, 발표 영상
- PR : [1주차 issue](https://github.com/sangjun19/Deobfuscator/issues/3)