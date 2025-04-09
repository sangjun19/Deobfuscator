	.file	"bvernu_LCD-Calculator_main_flatten.c"
	.text
	.globl	_TIG_IZ_T8jN_argc
	.bss
	.align 4
	.type	_TIG_IZ_T8jN_argc, @object
	.size	_TIG_IZ_T8jN_argc, 4
_TIG_IZ_T8jN_argc:
	.zero	4
	.globl	currentState
	.align 4
	.type	currentState, @object
	.size	currentState, 4
currentState:
	.zero	4
	.globl	keypad
	.align 16
	.type	keypad, @object
	.size	keypad, 16
keypad:
	.zero	16
	.globl	buildforA
	.type	buildforA, @object
	.size	buildforA, 1
buildforA:
	.zero	1
	.globl	B
	.align 4
	.type	B, @object
	.size	B, 4
B:
	.zero	4
	.globl	_TIG_IZ_T8jN_envp
	.align 8
	.type	_TIG_IZ_T8jN_envp, @object
	.size	_TIG_IZ_T8jN_envp, 8
_TIG_IZ_T8jN_envp:
	.zero	8
	.globl	_TIG_IZ_T8jN_argv
	.align 8
	.type	_TIG_IZ_T8jN_argv, @object
	.size	_TIG_IZ_T8jN_argv, 8
_TIG_IZ_T8jN_argv:
	.zero	8
	.globl	A
	.align 4
	.type	A, @object
	.size	A, 4
A:
	.zero	4
	.text
	.globl	Delay
	.type	Delay, @function
Delay:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L13:
	cmpq	$5, -8(%rbp)
	ja	.L14
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L14-.L4
	.long	.L15-.L4
	.text
.L7:
	addq	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L9
.L5:
	movq	$2, -8(%rbp)
	jmp	.L9
.L8:
	movl	-20(%rbp), %eax
	cltq
	cmpq	%rax, -16(%rbp)
	jnb	.L11
	movq	$1, -8(%rbp)
	jmp	.L9
.L11:
	movq	$5, -8(%rbp)
	jmp	.L9
.L6:
	movq	$0, -16(%rbp)
	movq	$0, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L13
.L15:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	Delay, .-Delay
	.globl	WriteToIR
	.type	WriteToIR, @function
WriteToIR:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$2, -8(%rbp)
.L22:
	cmpq	$2, -8(%rbp)
	je	.L17
	cmpq	$2, -8(%rbp)
	ja	.L24
	cmpq	$0, -8(%rbp)
	je	.L19
	cmpq	$1, -8(%rbp)
	jne	.L24
	jmp	.L23
.L19:
	movl	$1073763324, %edx
	movzbl	-20(%rbp), %eax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	andl	$4294967291, %eax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	andl	$4294967287, %eax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	orq	$16, %rax
	movq	%rax, (%rdx)
	movl	$1, %edi
	call	delay_ms
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	andl	$4294967279, %eax
	movq	%rax, (%rdx)
	movl	$3, %edi
	call	delay_ms
	movq	$1, -8(%rbp)
	jmp	.L21
.L17:
	movq	$0, -8(%rbp)
	jmp	.L21
.L24:
	nop
.L21:
	jmp	.L22
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	WriteToIR, .-WriteToIR
	.globl	delay_ms
	.type	delay_ms, @function
delay_ms:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L41:
	cmpq	$9, -8(%rbp)
	ja	.L42
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L34-.L28
	.long	.L33-.L28
	.long	.L32-.L28
	.long	.L42-.L28
	.long	.L31-.L28
	.long	.L30-.L28
	.long	.L42-.L28
	.long	.L42-.L28
	.long	.L29-.L28
	.long	.L43-.L28
	.text
.L31:
	addl	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L35
.L29:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L35
.L33:
	movl	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L35
.L30:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jnb	.L37
	movq	$8, -8(%rbp)
	jmp	.L35
.L37:
	movq	$9, -8(%rbp)
	jmp	.L35
.L34:
	cmpl	$2999, -12(%rbp)
	ja	.L39
	movq	$2, -8(%rbp)
	jmp	.L35
.L39:
	movq	$4, -8(%rbp)
	jmp	.L35
.L32:
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L35
.L42:
	nop
.L35:
	jmp	.L41
.L43:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	delay_ms, .-delay_ms
	.globl	changingState
	.type	changingState, @function
changingState:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$12, -8(%rbp)
.L62:
	movq	-8(%rbp), %rax
	subq	$4, %rax
	cmpq	$10, %rax
	ja	.L63
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L53-.L47
	.long	.L52-.L47
	.long	.L63-.L47
	.long	.L51-.L47
	.long	.L64-.L47
	.long	.L49-.L47
	.long	.L63-.L47
	.long	.L63-.L47
	.long	.L48-.L47
	.long	.L63-.L47
	.long	.L46-.L47
	.text
.L53:
	call	clearingtheDisplay
	movl	$1, currentState(%rip)
	movq	$8, -8(%rbp)
	jmp	.L54
.L46:
	movl	A(%rip), %edx
	movl	B(%rip), %eax
	imull	%edx, %eax
	movl	%eax, %edi
	call	giveanswer
	movl	$1, currentState(%rip)
	call	resetCalculator
	movq	$8, -8(%rbp)
	jmp	.L54
.L48:
	cmpl	$3, -20(%rbp)
	je	.L55
	cmpl	$3, -20(%rbp)
	ja	.L56
	cmpl	$2, -20(%rbp)
	je	.L57
	cmpl	$2, -20(%rbp)
	ja	.L56
	cmpl	$0, -20(%rbp)
	je	.L58
	cmpl	$1, -20(%rbp)
	je	.L59
	jmp	.L56
.L55:
	movq	$14, -8(%rbp)
	jmp	.L60
.L57:
	movq	$5, -8(%rbp)
	jmp	.L60
.L59:
	movq	$4, -8(%rbp)
	jmp	.L60
.L58:
	movq	$7, -8(%rbp)
	jmp	.L60
.L56:
	movq	$9, -8(%rbp)
	nop
.L60:
	jmp	.L54
.L49:
	movq	$8, -8(%rbp)
	jmp	.L54
.L52:
	call	clearingtheDisplay
	movb	$0, buildforA(%rip)
	movl	$2, currentState(%rip)
	movq	$8, -8(%rbp)
	jmp	.L54
.L51:
	call	resetCalculator
	movl	$1, currentState(%rip)
	movq	$8, -8(%rbp)
	jmp	.L54
.L63:
	nop
.L54:
	jmp	.L62
.L64:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	changingState, .-changingState
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, buildforA(%rip)
	nop
.L66:
	movl	$0, B(%rip)
	nop
.L67:
	movl	$0, A(%rip)
	nop
.L68:
	movl	$0, currentState(%rip)
	nop
.L69:
	movb	$49, keypad(%rip)
	movb	$50, 1+keypad(%rip)
	movb	$51, 2+keypad(%rip)
	movb	$65, 3+keypad(%rip)
	movb	$52, 4+keypad(%rip)
	movb	$53, 5+keypad(%rip)
	movb	$54, 6+keypad(%rip)
	movb	$66, 7+keypad(%rip)
	movb	$55, 8+keypad(%rip)
	movb	$56, 9+keypad(%rip)
	movb	$57, 10+keypad(%rip)
	movb	$67, 11+keypad(%rip)
	movb	$42, 12+keypad(%rip)
	movb	$48, 13+keypad(%rip)
	movb	$35, 14+keypad(%rip)
	movb	$68, 15+keypad(%rip)
	nop
.L70:
	movq	$0, _TIG_IZ_T8jN_envp(%rip)
	nop
.L71:
	movq	$0, _TIG_IZ_T8jN_argv(%rip)
	nop
.L72:
	movl	$0, _TIG_IZ_T8jN_argc(%rip)
	nop
	nop
.L73:
.L74:
#APP
# 312 "bvernu_LCD-Calculator_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-T8jN--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_T8jN_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_T8jN_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_T8jN_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L85:
	cmpq	$6, -8(%rbp)
	ja	.L86
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L77(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L77(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L77:
	.long	.L86-.L77
	.long	.L86-.L77
	.long	.L81-.L77
	.long	.L80-.L77
	.long	.L79-.L77
	.long	.L78-.L77
	.long	.L76-.L77
	.text
.L79:
	cmpb	$0, -10(%rbp)
	je	.L82
	movq	$6, -8(%rbp)
	jmp	.L84
.L82:
	movq	$2, -8(%rbp)
	jmp	.L84
.L80:
	movq	$5, -8(%rbp)
	jmp	.L84
.L76:
	movsbl	-10(%rbp), %eax
	movl	%eax, %edi
	call	processKeypadInput
	movq	$2, -8(%rbp)
	jmp	.L84
.L78:
	call	LCD_init
	call	KeyPad_Init
	movq	$2, -8(%rbp)
	jmp	.L84
.L81:
	call	Check_Keypad
	movb	%al, -9(%rbp)
	movzbl	-9(%rbp), %eax
	movb	%al, -10(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L84
.L86:
	nop
.L84:
	jmp	.L85
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.globl	processKeypadInput
	.type	processKeypadInput, @function
processKeypadInput:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$6, -8(%rbp)
.L109:
	cmpq	$9, -8(%rbp)
	ja	.L110
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L90(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L90(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L90:
	.long	.L97-.L90
	.long	.L96-.L90
	.long	.L110-.L90
	.long	.L95-.L90
	.long	.L111-.L90
	.long	.L93-.L90
	.long	.L92-.L90
	.long	.L110-.L90
	.long	.L91-.L90
	.long	.L89-.L90
	.text
.L91:
	movl	$0, %edi
	call	changingState
	movq	$4, -8(%rbp)
	jmp	.L99
.L96:
	movsbl	-20(%rbp), %eax
	movl	%eax, %edi
	call	takingInDig
	movq	$4, -8(%rbp)
	jmp	.L99
.L95:
	cmpb	$47, -20(%rbp)
	jle	.L100
	movq	$5, -8(%rbp)
	jmp	.L99
.L100:
	movq	$4, -8(%rbp)
	jmp	.L99
.L89:
	movl	$2, %edi
	call	changingState
	movq	$4, -8(%rbp)
	jmp	.L99
.L92:
	movsbl	-20(%rbp), %eax
	cmpl	$67, %eax
	je	.L102
	cmpl	$67, %eax
	jg	.L103
	cmpl	$35, %eax
	je	.L104
	cmpl	$42, %eax
	je	.L105
	jmp	.L103
.L104:
	movq	$0, -8(%rbp)
	jmp	.L106
.L105:
	movq	$9, -8(%rbp)
	jmp	.L106
.L102:
	movq	$8, -8(%rbp)
	jmp	.L106
.L103:
	movq	$3, -8(%rbp)
	nop
.L106:
	jmp	.L99
.L93:
	cmpb	$57, -20(%rbp)
	jg	.L107
	movq	$1, -8(%rbp)
	jmp	.L99
.L107:
	movq	$4, -8(%rbp)
	jmp	.L99
.L97:
	movl	$3, %edi
	call	changingState
	movq	$4, -8(%rbp)
	jmp	.L99
.L110:
	nop
.L99:
	jmp	.L109
.L111:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	processKeypadInput, .-processKeypadInput
	.globl	resetCalculator
	.type	resetCalculator, @function
resetCalculator:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L118:
	cmpq	$2, -8(%rbp)
	je	.L119
	cmpq	$2, -8(%rbp)
	ja	.L120
	cmpq	$0, -8(%rbp)
	je	.L115
	cmpq	$1, -8(%rbp)
	jne	.L120
	movl	$0, A(%rip)
	movl	$0, B(%rip)
	movb	$1, buildforA(%rip)
	call	clearingtheDisplay
	movq	$2, -8(%rbp)
	jmp	.L116
.L115:
	movq	$1, -8(%rbp)
	jmp	.L116
.L120:
	nop
.L116:
	jmp	.L118
.L119:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	resetCalculator, .-resetCalculator
	.globl	giveanswer
	.type	giveanswer, @function
giveanswer:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -148(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -120(%rbp)
.L150:
	cmpq	$26, -120(%rbp)
	ja	.L153
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L124(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L124(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L124:
	.long	.L153-.L124
	.long	.L139-.L124
	.long	.L138-.L124
	.long	.L154-.L124
	.long	.L136-.L124
	.long	.L135-.L124
	.long	.L134-.L124
	.long	.L153-.L124
	.long	.L153-.L124
	.long	.L153-.L124
	.long	.L153-.L124
	.long	.L153-.L124
	.long	.L133-.L124
	.long	.L132-.L124
	.long	.L131-.L124
	.long	.L130-.L124
	.long	.L129-.L124
	.long	.L153-.L124
	.long	.L128-.L124
	.long	.L153-.L124
	.long	.L127-.L124
	.long	.L153-.L124
	.long	.L126-.L124
	.long	.L153-.L124
	.long	.L125-.L124
	.long	.L153-.L124
	.long	.L123-.L124
	.text
.L128:
	movl	-140(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -128(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L140
.L136:
	movl	$5000000, %edi
	call	Delay
	movq	$3, -120(%rbp)
	jmp	.L140
.L131:
	movl	-140(%rbp), %eax
	cltq
	movb	$0, -112(%rbp,%rax)
	movl	$0, -124(%rbp)
	movq	$24, -120(%rbp)
	jmp	.L140
.L130:
	movl	-148(%rbp), %ecx
	movl	%ecx, %edx
	movl	$3435973837, %eax
	imulq	%rdx, %rax
	shrq	$32, %rax
	movl	%eax, %edx
	shrl	$3, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, %eax
	addl	$48, %eax
	movl	%eax, %edx
	movl	-128(%rbp), %eax
	cltq
	movb	%dl, -112(%rbp,%rax)
	movl	-148(%rbp), %eax
	movl	%eax, %edx
	movl	$3435973837, %eax
	imulq	%rdx, %rax
	shrq	$32, %rax
	shrl	$3, %eax
	movl	%eax, -148(%rbp)
	subl	$1, -128(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L140
.L133:
	cmpl	$15, -132(%rbp)
	jg	.L141
	movq	$22, -120(%rbp)
	jmp	.L140
.L141:
	movq	$18, -120(%rbp)
	jmp	.L140
.L139:
	cmpl	$0, -128(%rbp)
	js	.L143
	movq	$15, -120(%rbp)
	jmp	.L140
.L143:
	movq	$14, -120(%rbp)
	jmp	.L140
.L129:
	addl	$1, -140(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L140
.L125:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	testb	%al, %al
	je	.L146
	movq	$26, -120(%rbp)
	jmp	.L140
.L146:
	movq	$4, -120(%rbp)
	jmp	.L140
.L123:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	LCD_print
	addl	$1, -124(%rbp)
	movq	$24, -120(%rbp)
	jmp	.L140
.L132:
	call	clearingtheDisplay
	movl	$192, %edi
	call	WriteToIR
	movl	$1, -140(%rbp)
	movl	-148(%rbp), %eax
	movl	%eax, -136(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L140
.L134:
	movl	$0, -132(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L140
.L126:
	movl	-132(%rbp), %eax
	cltq
	movb	$48, -112(%rbp,%rax)
	addl	$1, -132(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L140
.L135:
	movl	-136(%rbp), %eax
	movl	%eax, %edx
	movl	$3435973837, %eax
	imulq	%rdx, %rax
	shrq	$32, %rax
	shrl	$3, %eax
	movl	%eax, -136(%rbp)
	movq	$20, -120(%rbp)
	jmp	.L140
.L138:
	movq	$13, -120(%rbp)
	jmp	.L140
.L127:
	cmpl	$0, -136(%rbp)
	je	.L148
	movq	$16, -120(%rbp)
	jmp	.L140
.L148:
	movq	$6, -120(%rbp)
	jmp	.L140
.L153:
	nop
.L140:
	jmp	.L150
.L154:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L152
	call	__stack_chk_fail@PLT
.L152:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	giveanswer, .-giveanswer
	.globl	clearingtheDisplay
	.type	clearingtheDisplay, @function
clearingtheDisplay:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L161:
	cmpq	$2, -8(%rbp)
	je	.L156
	cmpq	$2, -8(%rbp)
	ja	.L162
	cmpq	$0, -8(%rbp)
	je	.L163
	cmpq	$1, -8(%rbp)
	jne	.L162
	movq	$2, -8(%rbp)
	jmp	.L159
.L156:
	movl	$1, %edi
	call	WriteToIR
	movl	$2, %edi
	call	delay_ms
	movq	$0, -8(%rbp)
	jmp	.L159
.L162:
	nop
.L159:
	jmp	.L161
.L163:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	clearingtheDisplay, .-clearingtheDisplay
	.globl	Check_Keypad
	.type	Check_Keypad, @function
Check_Keypad:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L188:
	cmpq	$16, -8(%rbp)
	ja	.L189
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L167(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L167(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L167:
	.long	.L189-.L167
	.long	.L177-.L167
	.long	.L189-.L167
	.long	.L176-.L167
	.long	.L189-.L167
	.long	.L175-.L167
	.long	.L174-.L167
	.long	.L189-.L167
	.long	.L173-.L167
	.long	.L172-.L167
	.long	.L171-.L167
	.long	.L189-.L167
	.long	.L170-.L167
	.long	.L169-.L167
	.long	.L189-.L167
	.long	.L168-.L167
	.long	.L166-.L167
	.text
.L168:
	addl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L178
.L170:
	movl	-16(%rbp), %eax
	addl	$4, %eax
	movl	$1, %edx
	movl	%eax, %ecx
	sall	%cl, %edx
	movl	%edx, %eax
	movl	$1073767420, %edx
	cltq
	movq	%rax, (%rdx)
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L178
.L173:
	movl	$0, %eax
	jmp	.L179
.L177:
	movl	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L178
.L176:
	cmpl	$3, -12(%rbp)
	jg	.L180
	movq	$9, -8(%rbp)
	jmp	.L178
.L180:
	movq	$16, -8(%rbp)
	jmp	.L178
.L166:
	addl	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L178
.L172:
	movl	$1073890300, %eax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %ecx
	sall	%cl, %esi
	movl	%esi, %eax
	cltq
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L182
	movq	$10, -8(%rbp)
	jmp	.L178
.L182:
	movq	$15, -8(%rbp)
	jmp	.L178
.L169:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	leaq	keypad(%rip), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	jmp	.L179
.L174:
	movl	$50000, %edi
	call	Delay
	movq	$13, -8(%rbp)
	jmp	.L178
.L175:
	cmpl	$3, -16(%rbp)
	jg	.L184
	movq	$12, -8(%rbp)
	jmp	.L178
.L184:
	movq	$8, -8(%rbp)
	jmp	.L178
.L171:
	movl	$1073890300, %eax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %ecx
	sall	%cl, %esi
	movl	%esi, %eax
	cltq
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L186
	movq	$10, -8(%rbp)
	jmp	.L178
.L186:
	movq	$6, -8(%rbp)
	jmp	.L178
.L189:
	nop
.L178:
	jmp	.L188
.L179:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	Check_Keypad, .-Check_Keypad
	.globl	displayTopRow
	.type	displayTopRow, @function
displayTopRow:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$0, -8(%rbp)
.L195:
	cmpq	$0, -8(%rbp)
	je	.L191
	cmpq	$1, -8(%rbp)
	jne	.L197
	jmp	.L196
.L191:
	movsbl	-20(%rbp), %eax
	movl	%eax, %edi
	call	LCD_print
	movq	$1, -8(%rbp)
	jmp	.L194
.L197:
	nop
.L194:
	jmp	.L195
.L196:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	displayTopRow, .-displayTopRow
	.globl	KeyPad_Init
	.type	KeyPad_Init, @function
KeyPad_Init:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L204:
	cmpq	$2, -8(%rbp)
	je	.L205
	cmpq	$2, -8(%rbp)
	ja	.L206
	cmpq	$0, -8(%rbp)
	je	.L201
	cmpq	$1, -8(%rbp)
	jne	.L206
	movl	$1074783752, %eax
	movq	(%rax), %rax
	movl	$1074783752, %edx
	orq	$20, %rax
	movq	%rax, (%rdx)
	movl	$1073767424, %eax
	movq	(%rax), %rax
	movl	$1073767424, %edx
	orb	$-16, %al
	movq	%rax, (%rdx)
	movl	$1073890304, %eax
	movq	(%rax), %rax
	movl	$1073890304, %edx
	andq	$-16, %rax
	movq	%rax, (%rdx)
	movl	$1073890580, %eax
	movq	(%rax), %rax
	movl	$1073890580, %edx
	orq	$15, %rax
	movq	%rax, (%rdx)
	movl	$1073767708, %eax
	movq	(%rax), %rax
	movl	$1073767708, %edx
	orb	$-16, %al
	movq	%rax, (%rdx)
	movl	$1073890588, %eax
	movq	(%rax), %rax
	movl	$1073890588, %edx
	orq	$15, %rax
	movq	%rax, (%rdx)
	movq	$2, -8(%rbp)
	jmp	.L202
.L201:
	movq	$1, -8(%rbp)
	jmp	.L202
.L206:
	nop
.L202:
	jmp	.L204
.L205:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	KeyPad_Init, .-KeyPad_Init
	.globl	WriteToDR
	.type	WriteToDR, @function
WriteToDR:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$2, -8(%rbp)
.L213:
	cmpq	$2, -8(%rbp)
	je	.L208
	cmpq	$2, -8(%rbp)
	ja	.L215
	cmpq	$0, -8(%rbp)
	je	.L210
	cmpq	$1, -8(%rbp)
	jne	.L215
	jmp	.L214
.L210:
	movl	$1073763324, %edx
	movsbq	-20(%rbp), %rax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	orq	$4, %rax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	andl	$4294967287, %eax
	movq	%rax, (%rdx)
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	orq	$16, %rax
	movq	%rax, (%rdx)
	movl	$1, %edi
	call	delay_ms
	movl	$1073759228, %eax
	movq	(%rax), %rax
	movl	$1073759228, %edx
	andl	$4294967279, %eax
	movq	%rax, (%rdx)
	movl	$3, %edi
	call	delay_ms
	movq	$1, -8(%rbp)
	jmp	.L212
.L208:
	movq	$0, -8(%rbp)
	jmp	.L212
.L215:
	nop
.L212:
	jmp	.L213
.L214:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	WriteToDR, .-WriteToDR
	.globl	takingInDig
	.type	takingInDig, @function
takingInDig:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, %eax
	movb	%al, -36(%rbp)
	movq	$3, -16(%rbp)
.L228:
	cmpq	$4, -16(%rbp)
	ja	.L229
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L219(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L219(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L219:
	.long	.L223-.L219
	.long	.L230-.L219
	.long	.L221-.L219
	.long	.L220-.L219
	.long	.L218-.L219
	.text
.L218:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	(%rax), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	movl	%eax, %edx
	movsbl	-36(%rbp), %eax
	addl	%edx, %eax
	leal	-48(%rax), %edx
	movq	-8(%rbp), %rax
	movl	%edx, (%rax)
	movsbl	-36(%rbp), %eax
	movl	%eax, %edi
	call	displayTopRow
	movq	$1, -16(%rbp)
	jmp	.L224
.L220:
	movzbl	buildforA(%rip), %eax
	testb	%al, %al
	je	.L226
	movq	$2, -16(%rbp)
	jmp	.L224
.L226:
	movq	$0, -16(%rbp)
	jmp	.L224
.L223:
	leaq	B(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L224
.L221:
	leaq	A(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L224
.L229:
	nop
.L224:
	jmp	.L228
.L230:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	takingInDig, .-takingInDig
	.globl	LCD_init
	.type	LCD_init, @function
LCD_init:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L237:
	cmpq	$2, -8(%rbp)
	je	.L232
	cmpq	$2, -8(%rbp)
	ja	.L238
	cmpq	$0, -8(%rbp)
	je	.L239
	cmpq	$1, -8(%rbp)
	jne	.L238
	movq	$2, -8(%rbp)
	jmp	.L235
.L232:
	movl	$1074783752, %eax
	movq	(%rax), %rax
	movl	$1074783752, %edx
	orq	$3, %rax
	movq	%rax, (%rdx)
	movl	$100, %edi
	call	delay_ms
	movl	$1073759232, %eax
	movq	(%rax), %rax
	movl	$1073759232, %edx
	orq	$28, %rax
	movq	%rax, (%rdx)
	movl	$1073759516, %eax
	movq	(%rax), %rax
	movl	$1073759516, %edx
	orq	$28, %rax
	movq	%rax, (%rdx)
	movl	$1073763328, %eax
	movq	$255, (%rax)
	movl	$1073763612, %eax
	movq	$255, (%rax)
	movl	$56, %edi
	call	WriteToIR
	movl	$6, %edi
	call	WriteToIR
	movl	$15, %edi
	call	WriteToIR
	movl	$1, %edi
	call	WriteToIR
	movl	$3, %edi
	call	delay_ms
	movq	$0, -8(%rbp)
	jmp	.L235
.L238:
	nop
.L235:
	jmp	.L237
.L239:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	LCD_init, .-LCD_init
	.globl	LCD_print
	.type	LCD_print, @function
LCD_print:
.LFB20:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$2, -8(%rbp)
.L248:
	cmpq	$2, -8(%rbp)
	je	.L241
	cmpq	$2, -8(%rbp)
	ja	.L250
	cmpq	$0, -8(%rbp)
	je	.L243
	cmpq	$1, -8(%rbp)
	jne	.L250
	jmp	.L249
.L243:
	movsbl	-20(%rbp), %eax
	movl	%eax, %edi
	call	WriteToDR
	movq	$1, -8(%rbp)
	jmp	.L245
.L241:
	cmpb	$0, -20(%rbp)
	je	.L246
	movq	$0, -8(%rbp)
	jmp	.L245
.L246:
	movq	$1, -8(%rbp)
	jmp	.L245
.L250:
	nop
.L245:
	jmp	.L248
.L249:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	LCD_print, .-LCD_print
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
