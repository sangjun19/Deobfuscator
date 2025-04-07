	.file	"jesses-code-adventures_http-from-scratch_main_flatten.c"
	.text
	.globl	_TIG_IZ_cxZ4_envp
	.bss
	.align 8
	.type	_TIG_IZ_cxZ4_envp, @object
	.size	_TIG_IZ_cxZ4_envp, 8
_TIG_IZ_cxZ4_envp:
	.zero	8
	.globl	_TIG_IZ_cxZ4_argc
	.align 4
	.type	_TIG_IZ_cxZ4_argc, @object
	.size	_TIG_IZ_cxZ4_argc, 4
_TIG_IZ_cxZ4_argc:
	.zero	4
	.globl	_TIG_IZ_cxZ4_argv
	.align 8
	.type	_TIG_IZ_cxZ4_argv, @object
	.size	_TIG_IZ_cxZ4_argv, 8
_TIG_IZ_cxZ4_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Error creating socket\n"
	.align 8
.LC1:
	.string	"Error creating connection file descriptor\n"
.LC2:
	.string	"Error creating thread\n"
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
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$320, %rsp
	movl	%edi, -8484(%rbp)
	movq	%rsi, -8496(%rbp)
	movq	%rdx, -8504(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_cxZ4_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_cxZ4_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_cxZ4_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-cxZ4--0
# 0 "" 2
#NO_APP
	movl	-8484(%rbp), %eax
	movl	%eax, _TIG_IZ_cxZ4_argc(%rip)
	movq	-8496(%rbp), %rax
	movq	%rax, _TIG_IZ_cxZ4_argv(%rip)
	movq	-8504(%rbp), %rax
	movq	%rax, _TIG_IZ_cxZ4_envp(%rip)
	nop
	movq	$20, -8456(%rbp)
.L28:
	cmpq	$21, -8456(%rbp)
	ja	.L31
	movq	-8456(%rbp), %rax
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
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L31-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	leaq	-7616(%rbp), %rcx
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rax, %rcx
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rbp, %rax
	subq	$5616, %rax
	movq	%rcx, (%rax)
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rbp, %rax
	leaq	-5608(%rax), %rdx
	leaq	-3216(%rbp), %rax
	movq	%rax, (%rdx)
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rbp, %rax
	leaq	-5600(%rax), %rdx
	movl	-8464(%rbp), %eax
	movl	%eax, (%rdx)
	leaq	-5616(%rbp), %rcx
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	leaq	-8416(%rbp), %rcx
	movl	-8464(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rdx, %rcx
	leaq	handle_client(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	movl	%eax, -8460(%rbp)
	movq	$2, -8456(%rbp)
	jmp	.L20
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$6, -8456(%rbp)
	jmp	.L20
.L15:
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rbp, %rax
	subq	$7616, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L21
	movq	$21, -8456(%rbp)
	jmp	.L20
.L21:
	movq	$14, -8456(%rbp)
	jmp	.L20
.L18:
	cmpl	$-1, -8468(%rbp)
	jne	.L23
	movq	$15, -8456(%rbp)
	jmp	.L20
.L23:
	movq	$13, -8456(%rbp)
	jmp	.L20
.L11:
	movl	$16, -8472(%rbp)
	leaq	-8472(%rbp), %rdx
	leaq	-8448(%rbp), %rcx
	movl	-8468(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, %edx
	movl	-8464(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$2, %rax
	addq	%rbp, %rax
	subq	$7616, %rax
	movl	%edx, (%rax)
	movq	$12, -8456(%rbp)
	jmp	.L20
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$16, -8456(%rbp)
	jmp	.L20
.L14:
	movl	-8468(%rbp), %edx
	movq	-8432(%rbp), %rcx
	movq	-8424(%rbp), %rax
	movq	%rcx, %rdi
	movq	%rax, %rsi
	call	init_server
	movl	$0, -8464(%rbp)
	movq	$16, -8456(%rbp)
	jmp	.L20
.L10:
	movl	-8464(%rbp), %eax
	cltq
	movq	-8416(%rbp,%rax,8), %rax
	movq	%rax, %rdi
	call	pthread_detach@PLT
	movl	-8464(%rbp), %eax
	addl	$1, %eax
	movslq	%eax, %rdx
	imulq	$1374389535, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$5, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	movl	%edx, -8464(%rbp)
	movl	-8464(%rbp), %edx
	imull	$100, %edx, %edx
	subl	%edx, %eax
	movl	%eax, -8464(%rbp)
	movq	$16, -8456(%rbp)
	jmp	.L20
.L16:
	movl	$1, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L17:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-8464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rbp, %rax
	subq	$7616, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$16, -8456(%rbp)
	jmp	.L20
.L19:
	cmpl	$0, -8460(%rbp)
	je	.L26
	movq	$5, -8456(%rbp)
	jmp	.L20
.L26:
	movq	$17, -8456(%rbp)
	jmp	.L20
.L9:
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -8468(%rbp)
	movq	$3, -8456(%rbp)
	jmp	.L20
.L31:
	nop
.L20:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC3:
	.string	"HTTP/1.1 %d OK\r\nContent-Length: %ld\r\nContent-Type: %s\r\n\r\n%s"
	.text
	.globl	buffer_ok_response
	.type	buffer_ok_response, @function
buffer_ok_response:
.LFB3:
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
.L37:
	cmpq	$0, -8(%rbp)
	je	.L33
	cmpq	$1, -8(%rbp)
	jne	.L39
	movq	-32(%rbp), %rax
	jmp	.L38
.L33:
	movq	-24(%rbp), %rax
	movq	24(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	16(%rax), %rsi
	movq	-24(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	leaq	.LC3(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$1, -8(%rbp)
	jmp	.L36
.L39:
	nop
.L36:
	jmp	.L37
.L38:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	buffer_ok_response, .-buffer_ok_response
	.section	.rodata
.LC4:
	.string	"Error binding socket"
.LC5:
	.string	"127.0.0.1"
.LC6:
	.string	"Error listening on socket\n"
	.text
	.globl	init_server
	.type	init_server, @function
init_server:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, %rcx
	movq	%rsi, %rax
	movq	%rax, %rbx
	movq	%rcx, -48(%rbp)
	movq	%rbx, -40(%rbp)
	movl	%edx, -52(%rbp)
	movq	$11, -24(%rbp)
.L59:
	cmpq	$11, -24(%rbp)
	ja	.L60
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L43(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L43(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L43:
	.long	.L52-.L43
	.long	.L60-.L43
	.long	.L51-.L43
	.long	.L50-.L43
	.long	.L60-.L43
	.long	.L49-.L43
	.long	.L48-.L43
	.long	.L47-.L43
	.long	.L46-.L43
	.long	.L45-.L43
	.long	.L44-.L43
	.long	.L42-.L43
	.text
.L46:
	movl	$0, %eax
	jmp	.L53
.L50:
	movl	$1, %eax
	jmp	.L53
.L42:
	movq	$6, -24(%rbp)
	jmp	.L54
.L45:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-52(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$5, -24(%rbp)
	jmp	.L54
.L48:
	leaq	-48(%rbp), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$2, -48(%rbp)
	movl	$6000, %edi
	call	htons@PLT
	movw	%ax, -46(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	inet_addr@PLT
	movl	%eax, -44(%rbp)
	leaq	-48(%rbp), %rcx
	movl	-52(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	%eax, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L54
.L49:
	movl	$1, %eax
	jmp	.L53
.L44:
	cmpl	$-1, -32(%rbp)
	jne	.L55
	movq	$9, -24(%rbp)
	jmp	.L54
.L55:
	movq	$7, -24(%rbp)
	jmp	.L54
.L52:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-52(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$3, -24(%rbp)
	jmp	.L54
.L47:
	movl	-52(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	%eax, -28(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L54
.L51:
	cmpl	$-1, -28(%rbp)
	jne	.L57
	movq	$0, -24(%rbp)
	jmp	.L54
.L57:
	movq	$8, -24(%rbp)
	jmp	.L54
.L60:
	nop
.L54:
	jmp	.L59
.L53:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	init_server, .-init_server
	.globl	create_response
	.type	create_response, @function
create_response:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movl	%r8d, -40(%rbp)
	movq	$0, -24(%rbp)
.L71:
	cmpq	$4, -24(%rbp)
	je	.L72
	cmpq	$4, -24(%rbp)
	ja	.L73
	cmpq	$2, -24(%rbp)
	je	.L64
	cmpq	$2, -24(%rbp)
	ja	.L73
	cmpq	$0, -24(%rbp)
	je	.L65
	cmpq	$1, -24(%rbp)
	je	.L74
	jmp	.L73
.L65:
	cmpq	$0, -56(%rbp)
	jne	.L68
	movq	$4, -24(%rbp)
	jmp	.L70
.L68:
	movq	$2, -24(%rbp)
	jmp	.L70
.L64:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, 8(%rdx)
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-36(%rbp), %eax
	movl	%eax, (%rdx)
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movq	%rax, 16(%rdx)
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rax, 24(%rdx)
	movq	$1, -24(%rbp)
	jmp	.L70
.L73:
	nop
.L70:
	jmp	.L71
.L72:
	nop
	jmp	.L61
.L74:
	nop
.L61:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	create_response, .-create_response
	.section	.rodata
.LC7:
	.string	"Hello, World!"
.LC8:
	.string	"text/html"
.LC9:
	.string	"\n[client] %s"
.LC10:
	.string	"\nserver: %s"
	.text
	.globl	handle_client
	.type	handle_client, @function
handle_client:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2112, %rsp
	movq	%rdi, -2104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$14, -2072(%rbp)
.L93:
	cmpq	$16, -2072(%rbp)
	ja	.L96
	movq	-2072(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L96-.L78
	.long	.L86-.L78
	.long	.L96-.L78
	.long	.L85-.L78
	.long	.L84-.L78
	.long	.L96-.L78
	.long	.L96-.L78
	.long	.L83-.L78
	.long	.L96-.L78
	.long	.L82-.L78
	.long	.L81-.L78
	.long	.L96-.L78
	.long	.L96-.L78
	.long	.L80-.L78
	.long	.L79-.L78
	.long	.L96-.L78
	.long	.L77-.L78
	.text
.L84:
	movl	-2084(%rbp), %eax
	movb	$0, -1040(%rbp,%rax)
	addl	$1, -2084(%rbp)
	movq	$3, -2072(%rbp)
	jmp	.L87
.L79:
	movq	$9, -2072(%rbp)
	jmp	.L87
.L86:
	movq	-2080(%rbp), %rax
	movl	16(%rax), %edx
	movq	-2080(%rbp), %rax
	movq	8(%rax), %rax
	movl	%edx, %r8d
	movq	%rax, %rcx
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC8(%rip), %rax
	movq	%rax, %rsi
	movl	$200, %edi
	call	create_response
	movq	-2080(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-2080(%rbp), %rax
	movl	16(%rax), %eax
	cltq
	salq	$5, %rax
	addq	%rax, %rdx
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	buffer_ok_response
	leaq	-2064(%rbp), %rsi
	movl	-2092(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	recv@PLT
	leaq	-2064(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1040(%rbp), %rsi
	movl	-2092(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	send@PLT
	movl	-2092(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$16, -2072(%rbp)
	jmp	.L87
.L85:
	cmpl	$1023, -2084(%rbp)
	jbe	.L88
	movq	$1, -2072(%rbp)
	jmp	.L87
.L88:
	movq	$4, -2072(%rbp)
	jmp	.L87
.L77:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L94
	jmp	.L95
.L82:
	movq	-2104(%rbp), %rax
	movq	%rax, -2080(%rbp)
	movq	-2080(%rbp), %rax
	movq	(%rax), %rax
	movl	(%rax), %eax
	movl	%eax, -2092(%rbp)
	movb	$0, -2064(%rbp)
	movl	$1, -2088(%rbp)
	movq	$13, -2072(%rbp)
	jmp	.L87
.L80:
	cmpl	$1023, -2088(%rbp)
	jbe	.L91
	movq	$10, -2072(%rbp)
	jmp	.L87
.L91:
	movq	$7, -2072(%rbp)
	jmp	.L87
.L81:
	movb	$0, -1040(%rbp)
	movl	$1, -2084(%rbp)
	movq	$3, -2072(%rbp)
	jmp	.L87
.L83:
	movl	-2088(%rbp), %eax
	movb	$0, -2064(%rbp,%rax)
	addl	$1, -2088(%rbp)
	movq	$13, -2072(%rbp)
	jmp	.L87
.L96:
	nop
.L87:
	jmp	.L93
.L95:
	call	__stack_chk_fail@PLT
.L94:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	handle_client, .-handle_client
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
