	.file	"lebavui_ltm_20232_149495_domain2ip_flatten.c"
	.text
	.globl	_TIG_IZ_LQK6_argv
	.bss
	.align 8
	.type	_TIG_IZ_LQK6_argv, @object
	.size	_TIG_IZ_LQK6_argv, 8
_TIG_IZ_LQK6_argv:
	.zero	8
	.globl	_TIG_IZ_LQK6_argc
	.align 4
	.type	_TIG_IZ_LQK6_argc, @object
	.size	_TIG_IZ_LQK6_argc, 4
_TIG_IZ_LQK6_argc:
	.zero	4
	.globl	_TIG_IZ_LQK6_envp
	.align 8
	.type	_TIG_IZ_LQK6_envp, @object
	.size	_TIG_IZ_LQK6_envp, 8
_TIG_IZ_LQK6_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Khong phan giai duoc"
.LC1:
	.string	"IPv4"
.LC2:
	.string	"IP: %s\n"
.LC3:
	.string	"http"
.LC4:
	.string	"gmail.com"
.LC5:
	.string	"IPv6"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_LQK6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_LQK6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_LQK6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 151 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-LQK6--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_LQK6_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_LQK6_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_LQK6_envp(%rip)
	nop
	movq	$10, -48(%rbp)
.L31:
	cmpq	$14, -48(%rbp)
	ja	.L34
	movq	-48(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L34-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L34-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	cmpl	$0, -72(%rbp)
	je	.L21
	movq	$14, -48(%rbp)
	jmp	.L23
.L21:
	movq	$2, -48(%rbp)
	jmp	.L23
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -48(%rbp)
	jmp	.L23
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-56(%rbp), %rax
	movq	24(%rax), %rax
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	%rdx, -24(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	call	inet_ntoa@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -48(%rbp)
	jmp	.L23
.L14:
	movq	-56(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	$2, %eax
	jne	.L24
	movq	$12, -48(%rbp)
	jmp	.L23
.L24:
	movq	$5, -48(%rbp)
	jmp	.L23
.L19:
	movq	-56(%rbp), %rax
	movq	40(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$11, -48(%rbp)
	jmp	.L23
.L11:
	cmpq	$0, -56(%rbp)
	je	.L26
	movq	$8, -48(%rbp)
	jmp	.L23
.L26:
	movq	$6, -48(%rbp)
	jmp	.L23
.L13:
	leaq	-64(%rbp), %rax
	movq	%rax, %rcx
	movl	$0, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	getaddrinfo@PLT
	movl	%eax, -68(%rbp)
	movl	-68(%rbp), %eax
	movl	%eax, -72(%rbp)
	movq	$4, -48(%rbp)
	jmp	.L23
.L9:
	movl	$1, %eax
	jmp	.L32
.L15:
	movl	$0, %eax
	jmp	.L32
.L16:
	movq	-56(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	$10, %eax
	jne	.L29
	movq	$0, -48(%rbp)
	jmp	.L23
.L29:
	movq	$1, -48(%rbp)
	jmp	.L23
.L12:
	movq	$9, -48(%rbp)
	jmp	.L23
.L20:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -48(%rbp)
	jmp	.L23
.L18:
	movq	-64(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$11, -48(%rbp)
	jmp	.L23
.L34:
	nop
.L23:
	jmp	.L31
.L32:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
