	.file	"ornfelt_1brc_2_flatten.c"
	.text
	.globl	_TIG_IZ_1A55_argc
	.bss
	.align 4
	.type	_TIG_IZ_1A55_argc, @object
	.size	_TIG_IZ_1A55_argc, 4
_TIG_IZ_1A55_argc:
	.zero	4
	.globl	_TIG_IZ_1A55_envp
	.align 8
	.type	_TIG_IZ_1A55_envp, @object
	.size	_TIG_IZ_1A55_envp, 8
_TIG_IZ_1A55_envp:
	.zero	8
	.globl	_TIG_IZ_1A55_argv
	.align 8
	.type	_TIG_IZ_1A55_argv, @object
	.size	_TIG_IZ_1A55_argv, 8
_TIG_IZ_1A55_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"measurements.txt"
.LC1:
	.string	", "
.LC2:
	.string	"error opening file"
.LC3:
	.string	"\n"
.LC4:
	.string	"%s=%.1f/%.1f/%.1f%s"
.LC5:
	.string	""
.LC6:
	.string	"r"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-73728(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$1440, %rsp
	movl	%edi, -75140(%rbp)
	movq	%rsi, -75152(%rbp)
	movq	%rdx, -75160(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1A55_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1A55_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1A55_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 124 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1A55--0
# 0 "" 2
#NO_APP
	movl	-75140(%rbp), %eax
	movl	%eax, _TIG_IZ_1A55_argc(%rip)
	movq	-75152(%rbp), %rax
	movq	%rax, _TIG_IZ_1A55_argv(%rip)
	movq	-75160(%rbp), %rax
	movq	%rax, _TIG_IZ_1A55_envp(%rip)
	nop
	movq	$18, -75064(%rbp)
.L62:
	cmpq	$45, -75064(%rbp)
	ja	.L69
	movq	-75064(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L69-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L14-.L8
	.long	.L69-.L8
	.long	.L13-.L8
	.long	.L69-.L8
	.long	.L69-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L69-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L23:
	leaq	.LC0(%rip), %rax
	movq	%rax, -75104(%rbp)
	movq	$19, -75064(%rbp)
	jmp	.L38
.L33:
	leaq	.LC1(%rip), %rax
	movq	%rax, -75072(%rbp)
	movq	$22, -75064(%rbp)
	jmp	.L38
.L16:
	cmpl	$0, -75116(%rbp)
	jns	.L39
	movq	$41, -75064(%rbp)
	jmp	.L38
.L39:
	movq	$27, -75064(%rbp)
	jmp	.L38
.L26:
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58520, %rax
	movsd	-75088(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movq	$45, -75064(%rbp)
	jmp	.L38
.L15:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L27:
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58528, %rax
	movsd	-75088(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movq	$2, -75064(%rbp)
	jmp	.L38
.L31:
	movl	$0, -75128(%rbp)
	leaq	-75024(%rbp), %rax
	movl	$16384, %edx
	movl	$-1, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	$45, -75064(%rbp)
	jmp	.L38
.L7:
	movq	-75096(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -75080(%rbp)
	movq	$0, -75064(%rbp)
	jmp	.L38
.L36:
	movl	-75124(%rbp), %eax
	cltq
	movl	-75024(%rbp,%rax,4), %eax
	movl	%eax, -75116(%rbp)
	movq	$30, -75064(%rbp)
	jmp	.L38
.L20:
	movl	-75124(%rbp), %eax
	cltq
	movl	-75024(%rbp,%rax,4), %eax
	cmpl	$-1, %eax
	je	.L41
	movq	$11, -75064(%rbp)
	jmp	.L38
.L41:
	movq	$1, -75064(%rbp)
	jmp	.L38
.L34:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L63
	jmp	.L66
.L25:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-75096(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$3, -75064(%rbp)
	jmp	.L38
.L19:
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58528, %rax
	movsd	(%rax), %xmm0
	comisd	-75088(%rbp), %xmm0
	jbe	.L67
	movq	$12, -75064(%rbp)
	jmp	.L38
.L67:
	movq	$2, -75064(%rbp)
	jmp	.L38
.L28:
	movl	-75124(%rbp), %eax
	cltq
	movl	-75024(%rbp,%rax,4), %eax
	leaq	-58640(%rbp), %rdx
	cltq
	salq	$7, %rax
	addq	%rax, %rdx
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -75120(%rbp)
	movq	$7, -75064(%rbp)
	jmp	.L38
.L30:
	leaq	-1040(%rbp), %rax
	movl	$59, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -75048(%rbp)
	movq	-75048(%rbp), %rax
	movq	%rax, -75040(%rbp)
	movq	-75040(%rbp), %rax
	movb	$0, (%rax)
	movq	-75040(%rbp), %rax
	addq	$1, %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -75032(%rbp)
	movsd	-75032(%rbp), %xmm0
	movsd	%xmm0, -75088(%rbp)
	leaq	-1040(%rbp), %rdx
	movq	-75040(%rbp), %rax
	subq	%rdx, %rax
	movl	%eax, %edx
	leaq	-1040(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	hash
	movl	%eax, -75108(%rbp)
	movl	-75108(%rbp), %eax
	andl	$4095, %eax
	movl	%eax, -75124(%rbp)
	movq	$23, -75064(%rbp)
	jmp	.L38
.L22:
	cmpl	$1, -75140(%rbp)
	jle	.L47
	movq	$42, -75064(%rbp)
	jmp	.L38
.L47:
	movq	$10, -75064(%rbp)
	jmp	.L38
.L24:
	movl	-75124(%rbp), %eax
	addl	$1, %eax
	andl	$4095, %eax
	movl	%eax, -75124(%rbp)
	movq	$23, -75064(%rbp)
	jmp	.L38
.L12:
	cmpq	$0, -75096(%rbp)
	jne	.L49
	movq	$31, -75064(%rbp)
	jmp	.L38
.L49:
	movq	$8, -75064(%rbp)
	jmp	.L38
.L18:
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58536, %rax
	movsd	(%rax), %xmm0
	addsd	-75088(%rbp), %xmm0
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58536, %rax
	movsd	%xmm0, (%rax)
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58540, %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58540, %rax
	movl	%edx, (%rax)
	movq	$26, -75064(%rbp)
	jmp	.L38
.L21:
	movl	-75112(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58520, %rax
	movsd	(%rax), %xmm2
	movl	-75112(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58536, %rax
	movsd	(%rax), %xmm0
	movl	-75112(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58540, %rax
	movl	(%rax), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	divsd	%xmm1, %xmm0
	movl	-75112(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58528, %rax
	movq	(%rax), %rax
	leaq	-58640(%rbp), %rcx
	movl	-75112(%rbp), %edx
	movslq	%edx, %rdx
	salq	$7, %rdx
	addq	%rdx, %rcx
	movq	-75072(%rbp), %rdx
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	movq	%rcx, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	addl	$1, -75112(%rbp)
	movq	$44, -75064(%rbp)
	jmp	.L38
.L17:
	leaq	.LC5(%rip), %rax
	movq	%rax, -75072(%rbp)
	movq	$22, -75064(%rbp)
	jmp	.L38
.L9:
	movl	-75112(%rbp), %eax
	cmpl	-75128(%rbp), %eax
	jge	.L51
	movq	$37, -75064(%rbp)
	jmp	.L38
.L51:
	movq	$16, -75064(%rbp)
	jmp	.L38
.L13:
	movl	-75112(%rbp), %eax
	cmpl	-75128(%rbp), %eax
	jge	.L53
	movq	$4, -75064(%rbp)
	jmp	.L38
.L53:
	movq	$28, -75064(%rbp)
	jmp	.L38
.L11:
	leaq	-58640(%rbp), %rdx
	movl	-75128(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rax, %rdx
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	movl	-75128(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58536, %rax
	movsd	-75088(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movl	-75128(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58520, %rax
	movsd	-75088(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movl	-75128(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58528, %rax
	movsd	-75088(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movl	-75128(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58540, %rax
	movl	$1, (%rax)
	movl	-75124(%rbp), %eax
	cltq
	movl	-75128(%rbp), %edx
	movl	%edx, -75024(%rbp,%rax,4)
	addl	$1, -75128(%rbp)
	movq	$45, -75064(%rbp)
	jmp	.L38
.L29:
	movq	-75104(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -75056(%rbp)
	movq	-75056(%rbp), %rax
	movq	%rax, -75096(%rbp)
	movq	$40, -75064(%rbp)
	jmp	.L38
.L10:
	movq	-75152(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -75104(%rbp)
	movq	$10, -75064(%rbp)
	jmp	.L38
.L37:
	cmpq	$0, -75080(%rbp)
	je	.L55
	movq	$9, -75064(%rbp)
	jmp	.L38
.L55:
	movq	$35, -75064(%rbp)
	jmp	.L38
.L32:
	cmpl	$0, -75120(%rbp)
	je	.L57
	movq	$17, -75064(%rbp)
	jmp	.L38
.L57:
	movq	$1, -75064(%rbp)
	jmp	.L38
.L14:
	movl	-75128(%rbp), %eax
	movslq	%eax, %rsi
	leaq	-58640(%rbp), %rax
	leaq	cmp(%rip), %rdx
	movq	%rdx, %rcx
	movl	$128, %edx
	movq	%rax, %rdi
	call	qsort@PLT
	movl	$0, -75112(%rbp)
	movq	$44, -75064(%rbp)
	jmp	.L38
.L35:
	movl	-75116(%rbp), %eax
	cltq
	salq	$7, %rax
	addq	%rbp, %rax
	subq	$58520, %rax
	movsd	(%rax), %xmm1
	movsd	-75088(%rbp), %xmm0
	comisd	%xmm1, %xmm0
	jbe	.L68
	movq	$15, -75064(%rbp)
	jmp	.L38
.L68:
	movq	$45, -75064(%rbp)
	jmp	.L38
.L69:
	nop
.L38:
	jmp	.L62
.L66:
	call	__stack_chk_fail@PLT
.L63:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.type	hash, @function
hash:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$4, -8(%rbp)
.L82:
	cmpq	$7, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L73(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L73(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L73:
	.long	.L84-.L73
	.long	.L84-.L73
	.long	.L77-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L84-.L73
	.long	.L74-.L73
	.long	.L72-.L73
	.text
.L75:
	movq	$2, -8(%rbp)
	jmp	.L78
.L76:
	movl	-16(%rbp), %edx
	movl	%edx, %eax
	sall	$5, %eax
	subl	%edx, %eax
	movl	%eax, %ecx
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	addl	%ecx, %eax
	movl	%eax, -16(%rbp)
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L78
.L74:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L79
	movq	$3, -8(%rbp)
	jmp	.L78
.L79:
	movq	$7, -8(%rbp)
	jmp	.L78
.L72:
	movl	-16(%rbp), %eax
	jmp	.L83
.L77:
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L78
.L84:
	nop
.L78:
	jmp	.L82
.L83:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	hash, .-hash
	.type	cmp, @function
cmp:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$0, -8(%rbp)
.L90:
	cmpq	$0, -8(%rbp)
	je	.L86
	cmpq	$1, -8(%rbp)
	jne	.L92
	movl	-12(%rbp), %eax
	jmp	.L91
.L86:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L89
.L92:
	nop
.L89:
	jmp	.L90
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	cmp, .-cmp
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
