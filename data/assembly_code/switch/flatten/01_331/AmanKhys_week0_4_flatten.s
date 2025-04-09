	.file	"AmanKhys_week0_4_flatten.c"
	.text
	.globl	_TIG_IZ_1FRZ_envp
	.bss
	.align 8
	.type	_TIG_IZ_1FRZ_envp, @object
	.size	_TIG_IZ_1FRZ_envp, 8
_TIG_IZ_1FRZ_envp:
	.zero	8
	.globl	_TIG_IZ_1FRZ_argc
	.align 4
	.type	_TIG_IZ_1FRZ_argc, @object
	.size	_TIG_IZ_1FRZ_argc, 4
_TIG_IZ_1FRZ_argc:
	.zero	4
	.globl	_TIG_IZ_1FRZ_argv
	.align 8
	.type	_TIG_IZ_1FRZ_argv, @object
	.size	_TIG_IZ_1FRZ_argv, 8
_TIG_IZ_1FRZ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"length of the string: %i \n\n"
.LC1:
	.string	"enter the string:"
.LC2:
	.string	"%s"
	.text
	.globl	findStrLen
	.type	findStrLen, @function
findStrLen:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -120(%rbp)
.L14:
	cmpq	$7, -120(%rbp)
	ja	.L17
	movq	-120(%rbp), %rax
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
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L18-.L4
	.long	.L6-.L4
	.long	.L17-.L4
	.long	.L17-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	testb	%al, %al
	je	.L10
	movq	$3, -120(%rbp)
	jmp	.L12
.L10:
	movq	$6, -120(%rbp)
	jmp	.L12
.L6:
	addl	$1, -124(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L12
.L5:
	movl	-124(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -120(%rbp)
	jmp	.L12
.L9:
	movq	$7, -120(%rbp)
	jmp	.L12
.L3:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -124(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L12
.L17:
	nop
.L12:
	jmp	.L14
.L18:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L16
	call	__stack_chk_fail@PLT
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	findStrLen, .-findStrLen
	.section	.rodata
.LC3:
	.string	"reverse string: %s \n\n"
.LC4:
	.string	"enter the string: "
	.text
	.globl	reverseStr
	.type	reverseStr, @function
reverseStr:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$240, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -232(%rbp)
.L37:
	cmpq	$13, -232(%rbp)
	ja	.L40
	movq	-232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L40-.L22
	.long	.L30-.L22
	.long	.L40-.L22
	.long	.L40-.L22
	.long	.L29-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L40-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L24-.L22
	.long	.L23-.L22
	.long	.L40-.L22
	.long	.L41-.L22
	.text
.L29:
	movl	-240(%rbp), %eax
	subl	-236(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-236(%rbp), %eax
	cltq
	movzbl	-224(%rbp,%rax), %edx
	movslq	%ecx, %rax
	movb	%dl, -112(%rbp,%rax)
	subl	$1, -236(%rbp)
	movq	$11, -232(%rbp)
	jmp	.L31
.L26:
	movl	-240(%rbp), %eax
	cltq
	movzbl	-224(%rbp,%rax), %eax
	testb	%al, %al
	je	.L32
	movq	$9, -232(%rbp)
	jmp	.L31
.L32:
	movq	$5, -232(%rbp)
	jmp	.L31
.L30:
	movl	-240(%rbp), %eax
	cltq
	movb	$0, -112(%rbp,%rax)
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -232(%rbp)
	jmp	.L31
.L23:
	cmpl	$0, -236(%rbp)
	js	.L34
	movq	$4, -232(%rbp)
	jmp	.L31
.L34:
	movq	$1, -232(%rbp)
	jmp	.L31
.L25:
	addl	$1, -240(%rbp)
	movq	$8, -232(%rbp)
	jmp	.L31
.L27:
	movq	$10, -232(%rbp)
	jmp	.L31
.L28:
	movl	-240(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-240(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -236(%rbp)
	movq	$11, -232(%rbp)
	jmp	.L31
.L24:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -240(%rbp)
	movq	$8, -232(%rbp)
	jmp	.L31
.L40:
	nop
.L31:
	jmp	.L37
.L41:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L39
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	reverseStr, .-reverseStr
	.section	.rodata
.LC5:
	.string	"the concated string: %s \n\n"
.LC6:
	.string	"enter the two strings "
.LC7:
	.string	"- first string:"
.LC8:
	.string	"- second string:"
	.text
	.globl	concatStr
	.type	concatStr, @function
concatStr:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$336, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -328(%rbp)
.L59:
	cmpq	$13, -328(%rbp)
	ja	.L62
	movq	-328(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L62-.L45
	.long	.L62-.L45
	.long	.L63-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L62-.L45
	.long	.L62-.L45
	.long	.L62-.L45
	.long	.L46-.L45
	.long	.L62-.L45
	.long	.L44-.L45
	.text
.L50:
	movq	$7, -328(%rbp)
	jmp	.L53
.L51:
	movl	-336(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	testb	%al, %al
	je	.L54
	movq	$11, -328(%rbp)
	jmp	.L53
.L54:
	movq	$5, -328(%rbp)
	jmp	.L53
.L46:
	addl	$1, -336(%rbp)
	movq	$3, -328(%rbp)
	jmp	.L53
.L44:
	movl	-332(%rbp), %eax
	cltq
	movzbl	-320(%rbp,%rax), %edx
	movl	-336(%rbp), %eax
	cltq
	movb	%dl, -208(%rbp,%rax)
	addl	$1, -336(%rbp)
	addl	$1, -332(%rbp)
	movq	$5, -328(%rbp)
	jmp	.L53
.L48:
	movl	-336(%rbp), %eax
	cltq
	movb	$0, -208(%rbp,%rax)
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -328(%rbp)
	jmp	.L53
.L49:
	movl	-332(%rbp), %eax
	cltq
	movzbl	-320(%rbp,%rax), %eax
	testb	%al, %al
	je	.L56
	movq	$13, -328(%rbp)
	jmp	.L53
.L56:
	movq	$6, -328(%rbp)
	jmp	.L53
.L47:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-320(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -336(%rbp)
	movl	$0, -332(%rbp)
	movq	$3, -328(%rbp)
	jmp	.L53
.L62:
	nop
.L53:
	jmp	.L59
.L63:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L61
	call	__stack_chk_fail@PLT
.L61:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	concatStr, .-concatStr
	.section	.rodata
.LC9:
	.string	"exiting the program."
	.align 8
.LC10:
	.string	"enter the subsequent numbers for respective commands: \n"
.LC11:
	.string	"1 - string length"
.LC12:
	.string	"2 - string concat"
.LC13:
	.string	"3 - string reverse"
.LC14:
	.string	"enter the command: "
.LC15:
	.string	"exit"
.LC16:
	.string	"enter a valid command!! "
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1FRZ_envp(%rip)
	nop
.L65:
	movq	$0, _TIG_IZ_1FRZ_argv(%rip)
	nop
.L66:
	movl	$0, _TIG_IZ_1FRZ_argc(%rip)
	nop
	nop
.L67:
.L68:
#APP
# 88 "AmanKhys_week0_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1FRZ--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_1FRZ_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_1FRZ_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_1FRZ_envp(%rip)
	nop
	movq	$5, -120(%rbp)
.L90:
	cmpq	$14, -120(%rbp)
	ja	.L93
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L71(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L71(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L71:
	.long	.L80-.L71
	.long	.L79-.L71
	.long	.L78-.L71
	.long	.L93-.L71
	.long	.L93-.L71
	.long	.L77-.L71
	.long	.L76-.L71
	.long	.L75-.L71
	.long	.L93-.L71
	.long	.L74-.L71
	.long	.L93-.L71
	.long	.L93-.L71
	.long	.L73-.L71
	.long	.L72-.L71
	.long	.L70-.L71
	.text
.L70:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -120(%rbp)
	jmp	.L81
.L73:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-112(%rbp), %rax
	leaq	.LC15(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -124(%rbp)
	movq	$9, -120(%rbp)
	jmp	.L81
.L79:
	call	findStrLen
	movq	$12, -120(%rbp)
	jmp	.L81
.L74:
	cmpl	$0, -124(%rbp)
	jne	.L82
	movq	$14, -120(%rbp)
	jmp	.L81
.L82:
	movq	$13, -120(%rbp)
	jmp	.L81
.L72:
	movzbl	-112(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$51, %eax
	je	.L84
	cmpl	$51, %eax
	jg	.L85
	cmpl	$49, %eax
	je	.L86
	cmpl	$50, %eax
	je	.L87
	jmp	.L85
.L84:
	movq	$2, -120(%rbp)
	jmp	.L88
.L87:
	movq	$6, -120(%rbp)
	jmp	.L88
.L86:
	movq	$1, -120(%rbp)
	jmp	.L88
.L85:
	movq	$0, -120(%rbp)
	nop
.L88:
	jmp	.L81
.L76:
	call	concatStr
	movq	$12, -120(%rbp)
	jmp	.L81
.L77:
	movq	$12, -120(%rbp)
	jmp	.L81
.L80:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -120(%rbp)
	jmp	.L81
.L75:
	movl	$1, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L91
	jmp	.L92
.L78:
	call	reverseStr
	movq	$12, -120(%rbp)
	jmp	.L81
.L93:
	nop
.L81:
	jmp	.L90
.L92:
	call	__stack_chk_fail@PLT
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
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
