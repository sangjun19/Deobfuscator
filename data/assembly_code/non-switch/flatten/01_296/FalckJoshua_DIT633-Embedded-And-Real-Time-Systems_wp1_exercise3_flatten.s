	.file	"FalckJoshua_DIT633-Embedded-And-Real-Time-Systems_wp1_exercise3_flatten.c"
	.text
	.globl	ACTIVE_GAME
	.bss
	.align 4
	.type	ACTIVE_GAME, @object
	.size	ACTIVE_GAME, 4
ACTIVE_GAME:
	.zero	4
	.globl	GUESS_COUNTER
	.align 4
	.type	GUESS_COUNTER, @object
	.size	GUESS_COUNTER, 4
GUESS_COUNTER:
	.zero	4
	.globl	_TIG_IZ_spxw_argc
	.align 4
	.type	_TIG_IZ_spxw_argc, @object
	.size	_TIG_IZ_spxw_argc, 4
_TIG_IZ_spxw_argc:
	.zero	4
	.globl	MAX_GUESSES
	.align 4
	.type	MAX_GUESSES, @object
	.size	MAX_GUESSES, 4
MAX_GUESSES:
	.zero	4
	.globl	_TIG_IZ_spxw_argv
	.align 8
	.type	_TIG_IZ_spxw_argv, @object
	.size	_TIG_IZ_spxw_argv, 8
_TIG_IZ_spxw_argv:
	.zero	8
	.globl	ACTUAL_NUMBER
	.align 4
	.type	ACTUAL_NUMBER, @object
	.size	ACTUAL_NUMBER, 4
ACTUAL_NUMBER:
	.zero	4
	.globl	_TIG_IZ_spxw_envp
	.align 8
	.type	_TIG_IZ_spxw_envp, @object
	.size	_TIG_IZ_spxw_envp, 8
_TIG_IZ_spxw_envp:
	.zero	8
	.globl	GUESSED_NUMBER
	.align 4
	.type	GUESSED_NUMBER, @object
	.size	GUESSED_NUMBER, 4
GUESSED_NUMBER:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Would you like to play again Y/N "
.LC1:
	.string	" %c"
	.align 8
.LC2:
	.string	"Please enter a number between 1 and 100!"
.LC3:
	.string	"Give us your best guess!"
	.align 8
.LC4:
	.string	"Number is higher! Next time you'll get it :)"
.LC5:
	.string	"-- %d guesses left! --\n"
.LC6:
	.string	"Please enter a number!"
.LC7:
	.string	"%s"
.LC8:
	.string	"You ran out of guesses! :("
.LC9:
	.string	"The number was %d\n"
	.align 8
.LC10:
	.string	"Number is lower! You got it next time :)"
.LC11:
	.string	"You guessed correctly! :)"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$1, ACTIVE_GAME(%rip)
	nop
.L2:
	movl	$10, MAX_GUESSES(%rip)
	nop
.L3:
	movl	$1, GUESS_COUNTER(%rip)
	nop
.L4:
	movl	$0, GUESSED_NUMBER(%rip)
	nop
.L5:
	movl	$0, ACTUAL_NUMBER(%rip)
	nop
.L6:
	movq	$0, _TIG_IZ_spxw_envp(%rip)
	nop
.L7:
	movq	$0, _TIG_IZ_spxw_argv(%rip)
	nop
.L8:
	movl	$0, _TIG_IZ_spxw_argc(%rip)
	nop
	nop
.L9:
.L10:
#APP
# 145 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-spxw--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_spxw_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_spxw_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_spxw_envp(%rip)
	nop
	movq	$27, -128(%rbp)
.L59:
	cmpq	$37, -128(%rbp)
	ja	.L62
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L13(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L13(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L13:
	.long	.L62-.L13
	.long	.L36-.L13
	.long	.L35-.L13
	.long	.L34-.L13
	.long	.L33-.L13
	.long	.L32-.L13
	.long	.L62-.L13
	.long	.L62-.L13
	.long	.L62-.L13
	.long	.L62-.L13
	.long	.L31-.L13
	.long	.L30-.L13
	.long	.L29-.L13
	.long	.L62-.L13
	.long	.L28-.L13
	.long	.L62-.L13
	.long	.L27-.L13
	.long	.L26-.L13
	.long	.L25-.L13
	.long	.L62-.L13
	.long	.L24-.L13
	.long	.L23-.L13
	.long	.L22-.L13
	.long	.L62-.L13
	.long	.L62-.L13
	.long	.L62-.L13
	.long	.L21-.L13
	.long	.L20-.L13
	.long	.L19-.L13
	.long	.L18-.L13
	.long	.L62-.L13
	.long	.L17-.L13
	.long	.L16-.L13
	.long	.L62-.L13
	.long	.L15-.L13
	.long	.L14-.L13
	.long	.L62-.L13
	.long	.L12-.L13
	.text
.L25:
	movl	GUESS_COUNTER(%rip), %edx
	movl	MAX_GUESSES(%rip), %eax
	cmpl	%eax, %edx
	jle	.L37
	movq	$5, -128(%rbp)
	jmp	.L39
.L37:
	movq	$12, -128(%rbp)
	jmp	.L39
.L33:
	movl	GUESSED_NUMBER(%rip), %edx
	movl	ACTUAL_NUMBER(%rip), %eax
	cmpl	%eax, %edx
	jle	.L40
	movq	$10, -128(%rbp)
	jmp	.L39
.L40:
	movq	$37, -128(%rbp)
	jmp	.L39
.L28:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-141(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -128(%rbp)
	jmp	.L39
.L17:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -128(%rbp)
	jmp	.L39
.L29:
	movl	ACTIVE_GAME(%rip), %eax
	testl	%eax, %eax
	jne	.L42
	movq	$14, -128(%rbp)
	jmp	.L39
.L42:
	movq	$3, -128(%rbp)
	jmp	.L39
.L36:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	call	rand@PLT
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1374389535, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$5, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	imull	$100, %edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	leal	1(%rdx), %eax
	movl	%eax, ACTUAL_NUMBER(%rip)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -128(%rbp)
	jmp	.L39
.L34:
	movl	ACTUAL_NUMBER(%rip), %edx
	movl	GUESSED_NUMBER(%rip), %eax
	cmpl	%eax, %edx
	je	.L44
	movq	$34, -128(%rbp)
	jmp	.L39
.L44:
	movq	$21, -128(%rbp)
	jmp	.L39
.L27:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, GUESSED_NUMBER(%rip)
	movq	$35, -128(%rbp)
	jmp	.L39
.L23:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L60
	jmp	.L61
.L21:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	MAX_GUESSES(%rip), %eax
	movl	GUESS_COUNTER(%rip), %edx
	subl	%edx, %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	GUESS_COUNTER(%rip), %eax
	addl	$1, %eax
	movl	%eax, GUESS_COUNTER(%rip)
	movq	$18, -128(%rbp)
	jmp	.L39
.L30:
	cmpl	$0, -140(%rbp)
	jne	.L47
	movq	$17, -128(%rbp)
	jmp	.L39
.L47:
	movq	$16, -128(%rbp)
	jmp	.L39
.L16:
	movl	GUESSED_NUMBER(%rip), %eax
	cmpl	$100, %eax
	jle	.L49
	movq	$31, -128(%rbp)
	jmp	.L39
.L49:
	movq	$4, -128(%rbp)
	jmp	.L39
.L26:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -128(%rbp)
	jmp	.L39
.L20:
	movq	$1, -128(%rbp)
	jmp	.L39
.L15:
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	charIsNumber
	movl	%eax, -140(%rbp)
	movq	$11, -128(%rbp)
	jmp	.L39
.L22:
	movl	GUESSED_NUMBER(%rip), %edx
	movl	ACTUAL_NUMBER(%rip), %eax
	cmpl	%eax, %edx
	jne	.L51
	movq	$20, -128(%rbp)
	jmp	.L39
.L51:
	movq	$18, -128(%rbp)
	jmp	.L39
.L19:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -128(%rbp)
	jmp	.L39
.L32:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	ACTUAL_NUMBER(%rip), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, ACTIVE_GAME(%rip)
	movq	$12, -128(%rbp)
	jmp	.L39
.L12:
	movl	GUESSED_NUMBER(%rip), %edx
	movl	ACTUAL_NUMBER(%rip), %eax
	cmpl	%eax, %edx
	jge	.L53
	movq	$26, -128(%rbp)
	jmp	.L39
.L53:
	movq	$22, -128(%rbp)
	jmp	.L39
.L31:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	MAX_GUESSES(%rip), %eax
	movl	GUESS_COUNTER(%rip), %edx
	subl	%edx, %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	GUESS_COUNTER(%rip), %eax
	addl	$1, %eax
	movl	%eax, GUESS_COUNTER(%rip)
	movq	$18, -128(%rbp)
	jmp	.L39
.L14:
	movl	GUESSED_NUMBER(%rip), %eax
	testl	%eax, %eax
	jg	.L55
	movq	$28, -128(%rbp)
	jmp	.L39
.L55:
	movq	$32, -128(%rbp)
	jmp	.L39
.L18:
	movl	$1, ACTIVE_GAME(%rip)
	movl	$1, GUESS_COUNTER(%rip)
	call	rand@PLT
	movl	%eax, -136(%rbp)
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1374389535, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$5, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	imull	$100, %edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	leal	1(%rdx), %eax
	movl	%eax, ACTUAL_NUMBER(%rip)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -128(%rbp)
	jmp	.L39
.L35:
	movzbl	-141(%rbp), %eax
	cmpb	$89, %al
	jne	.L57
	movq	$29, -128(%rbp)
	jmp	.L39
.L57:
	movq	$21, -128(%rbp)
	jmp	.L39
.L24:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, ACTIVE_GAME(%rip)
	movq	$18, -128(%rbp)
	jmp	.L39
.L62:
	nop
.L39:
	jmp	.L59
.L61:
	call	__stack_chk_fail@PLT
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	charIsNumber
	.type	charIsNumber, @function
charIsNumber:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$8, -8(%rbp)
.L82:
	cmpq	$8, -8(%rbp)
	ja	.L83
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L73-.L66
	.long	.L72-.L66
	.long	.L71-.L66
	.long	.L70-.L66
	.long	.L69-.L66
	.long	.L68-.L66
	.long	.L83-.L66
	.long	.L67-.L66
	.long	.L65-.L66
	.text
.L69:
	movl	$1, %eax
	jmp	.L74
.L65:
	movq	$5, -8(%rbp)
	jmp	.L75
.L72:
	movl	$0, %eax
	jmp	.L74
.L70:
	movl	$0, %eax
	jmp	.L74
.L68:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L76
	movq	$7, -8(%rbp)
	jmp	.L75
.L76:
	movq	$4, -8(%rbp)
	jmp	.L75
.L73:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$57, %al
	jle	.L78
	movq	$3, -8(%rbp)
	jmp	.L75
.L78:
	movq	$2, -8(%rbp)
	jmp	.L75
.L67:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jg	.L80
	movq	$1, -8(%rbp)
	jmp	.L75
.L80:
	movq	$0, -8(%rbp)
	jmp	.L75
.L71:
	addq	$1, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L75
.L83:
	nop
.L75:
	jmp	.L82
.L74:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	charIsNumber, .-charIsNumber
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
