	.file	"PENGLaurine_LAB2_TFTP_main_flatten.c"
	.text
	.globl	_TIG_IZ_mlca_argv
	.bss
	.align 8
	.type	_TIG_IZ_mlca_argv, @object
	.size	_TIG_IZ_mlca_argv, 8
_TIG_IZ_mlca_argv:
	.zero	8
	.globl	_TIG_IZ_mlca_envp
	.align 8
	.type	_TIG_IZ_mlca_envp, @object
	.size	_TIG_IZ_mlca_envp, 8
_TIG_IZ_mlca_envp:
	.zero	8
	.globl	_TIG_IZ_mlca_argc
	.align 4
	.type	_TIG_IZ_mlca_argc, @object
	.size	_TIG_IZ_mlca_argc, 4
_TIG_IZ_mlca_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Error: read\n"
.LC1:
	.string	"Error: number of arguments\n"
.LC2:
	.string	"addr ip : %s "
.LC3:
	.string	"%s\n"
	.align 8
.LC4:
	.string	"addrinfo:\n--family: %d\n--socktype: %d\n--protocol: %d\n\n"
.LC5:
	.string	"unknown"
.LC6:
	.string	"IPv6"
.LC7:
	.string	"Error: getaddrinfo failure\n"
.LC8:
	.string	"octet"
.LC9:
	.string	"IPv4"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2256, %rsp
	movl	%edi, -2228(%rbp)
	movq	%rsi, -2240(%rbp)
	movq	%rdx, -2248(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_mlca_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_mlca_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_mlca_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 143 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-mlca--0
# 0 "" 2
#NO_APP
	movl	-2228(%rbp), %eax
	movl	%eax, _TIG_IZ_mlca_argc(%rip)
	movq	-2240(%rbp), %rax
	movq	%rax, _TIG_IZ_mlca_argv(%rip)
	movq	-2248(%rbp), %rax
	movq	%rax, _TIG_IZ_mlca_envp(%rip)
	nop
	movq	$1, -2160(%rbp)
.L57:
	cmpq	$48, -2160(%rbp)
	ja	.L60
	movq	-2160(%rbp), %rax
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
	.long	.L60-.L8
	.long	.L35-.L8
	.long	.L60-.L8
	.long	.L34-.L8
	.long	.L60-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L60-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L60-.L8
	.long	.L29-.L8
	.long	.L60-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L60-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L60-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L60-.L8
	.long	.L10-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L24:
	movl	-2200(%rbp), %eax
	cltq
	movb	$0, -1040(%rbp,%rax)
	leaq	-1040(%rbp), %rax
	movl	$1024, %edx
	movq	%rax, %rsi
	movl	$1, %edi
	call	write@PLT
	movq	$20, -2160(%rbp)
	jmp	.L36
.L18:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$12, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L16:
	leaq	-2064(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2144(%rbp)
	movq	-2144(%rbp), %rdx
	leaq	-2064(%rbp), %rcx
	movl	-2220(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	-1040(%rbp), %rsi
	movl	-2220(%rbp), %eax
	movl	$0, %ecx
	movl	$1023, %edx
	movl	%eax, %edi
	call	recv@PLT
	movq	%rax, -2136(%rbp)
	movq	-2136(%rbp), %rax
	movl	%eax, -2200(%rbp)
	movq	$9, -2160(%rbp)
	jmp	.L36
.L27:
	movq	-2192(%rbp), %rax
	movq	%rax, %rdi
	call	freeaddrinfo@PLT
	movb	$0, -2064(%rbp)
	movb	$1, -2063(%rbp)
	movl	$0, -2208(%rbp)
	movq	$17, -2160(%rbp)
	jmp	.L36
.L26:
	cmpl	$-1, -2212(%rbp)
	je	.L37
	movq	$14, -2160(%rbp)
	jmp	.L36
.L37:
	movq	$24, -2160(%rbp)
	jmp	.L36
.L15:
	cmpl	$-1, -2220(%rbp)
	jne	.L39
	movq	$29, -2160(%rbp)
	jmp	.L36
.L39:
	movq	$5, -2160(%rbp)
	jmp	.L36
.L31:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$27, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L35:
	cmpl	$3, -2228(%rbp)
	je	.L41
	movq	$8, -2160(%rbp)
	jmp	.L36
.L41:
	movq	$35, -2160(%rbp)
	jmp	.L36
.L20:
	movq	-2176(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -2128(%rbp)
	movq	-2176(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, %edx
	movq	-2128(%rbp), %rax
	movw	%dx, (%rax)
	movl	$1069, %edi
	call	htons@PLT
	movq	-2128(%rbp), %rdx
	movw	%ax, 2(%rdx)
	movq	stdout(%rip), %rax
	movq	-2168(%rbp), %rdx
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-2128(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, %edi
	call	inet_ntoa@PLT
	movq	%rax, -2120(%rbp)
	movq	stdout(%rip), %rax
	movq	-2120(%rbp), %rdx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-2176(%rbp), %rax
	movl	12(%rax), %esi
	movq	-2176(%rbp), %rax
	movl	8(%rax), %ecx
	movq	-2176(%rbp), %rax
	movl	4(%rax), %edx
	movq	stdout(%rip), %rax
	movl	%esi, %r8d
	leaq	.LC4(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-2176(%rbp), %rax
	movl	12(%rax), %edx
	movq	-2176(%rbp), %rax
	movl	8(%rax), %ecx
	movq	-2176(%rbp), %rax
	movl	4(%rax), %eax
	movl	%ecx, %esi
	movl	%eax, %edi
	call	socket@PLT
	movl	%eax, -2220(%rbp)
	movq	$31, -2160(%rbp)
	jmp	.L36
.L34:
	movl	-2208(%rbp), %eax
	movslq	%eax, %rdx
	movq	-2184(%rbp), %rax
	addq	%rdx, %rax
	movl	-2208(%rbp), %edx
	leal	2(%rdx), %ecx
	movzbl	(%rax), %edx
	movslq	%ecx, %rax
	movb	%dl, -2064(%rbp,%rax)
	addl	$1, -2208(%rbp)
	movq	$17, -2160(%rbp)
	jmp	.L36
.L19:
	movl	-2220(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$29, -2160(%rbp)
	jmp	.L36
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, -2168(%rbp)
	movq	$23, -2160(%rbp)
	jmp	.L36
.L29:
	movl	$0, -2204(%rbp)
	movq	$13, -2160(%rbp)
	jmp	.L36
.L30:
	cmpl	$0, -2200(%rbp)
	jns	.L43
	movq	$25, -2160(%rbp)
	jmp	.L36
.L43:
	movq	$18, -2160(%rbp)
	jmp	.L36
.L28:
	movl	-2204(%rbp), %eax
	cmpl	$7, %eax
	ja	.L45
	movq	$6, -2160(%rbp)
	jmp	.L36
.L45:
	movq	$30, -2160(%rbp)
	jmp	.L36
.L23:
	leaq	.LC6(%rip), %rax
	movq	%rax, -2168(%rbp)
	movq	$23, -2160(%rbp)
	jmp	.L36
.L25:
	movl	-2208(%rbp), %eax
	cmpl	$7, %eax
	ja	.L47
	movq	$3, -2160(%rbp)
	jmp	.L36
.L47:
	movq	$11, -2160(%rbp)
	jmp	.L36
.L10:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$27, %edx
	movl	$1, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L32:
	movl	-2204(%rbp), %eax
	cltq
	leaq	.LC8(%rip), %rdx
	addq	%rdx, %rax
	movl	-2208(%rbp), %ecx
	movl	-2204(%rbp), %edx
	addl	%ecx, %edx
	leal	2(%rdx), %ecx
	movzbl	(%rax), %edx
	movslq	%ecx, %rax
	movb	%dl, -2064(%rbp,%rax)
	addl	$1, -2204(%rbp)
	movq	$13, -2160(%rbp)
	jmp	.L36
.L11:
	movq	-2176(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	$2, %eax
	je	.L49
	cmpl	$10, %eax
	jne	.L50
	movq	$19, -2160(%rbp)
	jmp	.L51
.L49:
	movq	$48, -2160(%rbp)
	jmp	.L51
.L50:
	movq	$36, -2160(%rbp)
	nop
.L51:
	jmp	.L36
.L7:
	leaq	.LC9(%rip), %rax
	movq	%rax, -2168(%rbp)
	movq	$23, -2160(%rbp)
	jmp	.L36
.L21:
	cmpq	$0, -2176(%rbp)
	je	.L52
	movq	$38, -2160(%rbp)
	jmp	.L36
.L52:
	movq	$14, -2160(%rbp)
	jmp	.L36
.L9:
	movq	-2192(%rbp), %rax
	movq	%rax, -2176(%rbp)
	movq	$22, -2160(%rbp)
	jmp	.L36
.L33:
	movq	-2176(%rbp), %rax
	movl	16(%rax), %edx
	movq	-2176(%rbp), %rax
	movq	24(%rax), %rcx
	movl	-2220(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	connect@PLT
	movl	%eax, -2212(%rbp)
	movq	$15, -2160(%rbp)
	jmp	.L36
.L12:
	cmpl	$0, -2216(%rbp)
	je	.L54
	movq	$40, -2160(%rbp)
	jmp	.L36
.L54:
	movq	$47, -2160(%rbp)
	jmp	.L36
.L14:
	movq	-2240(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -2152(%rbp)
	movq	-2240(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -2184(%rbp)
	leaq	-2112(%rbp), %rax
	movl	$48, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movl	$2, -2108(%rbp)
	movl	$1, -2104(%rbp)
	movl	$6, -2100(%rbp)
	leaq	-2192(%rbp), %rcx
	leaq	-2112(%rbp), %rdx
	movq	-2152(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	getaddrinfo@PLT
	movl	%eax, -2196(%rbp)
	movl	-2196(%rbp), %eax
	movl	%eax, -2216(%rbp)
	movq	$37, -2160(%rbp)
	jmp	.L36
.L17:
	movq	-2176(%rbp), %rax
	movq	40(%rax), %rax
	movq	%rax, -2176(%rbp)
	movq	$22, -2160(%rbp)
	jmp	.L36
.L22:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L60:
	nop
.L36:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
