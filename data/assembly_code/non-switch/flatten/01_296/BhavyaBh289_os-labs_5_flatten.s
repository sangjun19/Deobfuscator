	.file	"BhavyaBh289_os-labs_5_flatten.c"
	.text
	.globl	_TIG_IZ_XFpM_argv
	.bss
	.align 8
	.type	_TIG_IZ_XFpM_argv, @object
	.size	_TIG_IZ_XFpM_argv, 8
_TIG_IZ_XFpM_argv:
	.zero	8
	.globl	out
	.align 4
	.type	out, @object
	.size	out, 4
out:
	.zero	4
	.globl	full
	.align 32
	.type	full, @object
	.size	full, 32
full:
	.zero	32
	.globl	mutex
	.align 32
	.type	mutex, @object
	.size	mutex, 40
mutex:
	.zero	40
	.globl	_TIG_IZ_XFpM_envp
	.align 8
	.type	_TIG_IZ_XFpM_envp, @object
	.size	_TIG_IZ_XFpM_envp, 8
_TIG_IZ_XFpM_envp:
	.zero	8
	.globl	in
	.align 4
	.type	in, @object
	.size	in, 4
in:
	.zero	4
	.globl	_TIG_IZ_XFpM_argc
	.align 4
	.type	_TIG_IZ_XFpM_argc, @object
	.size	_TIG_IZ_XFpM_argc, 4
_TIG_IZ_XFpM_argc:
	.zero	4
	.globl	empty
	.align 32
	.type	empty, @object
	.size	empty, 32
empty:
	.zero	32
	.globl	buffer
	.align 16
	.type	buffer, @object
	.size	buffer, 20
buffer:
	.zero	20
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
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, mutex(%rip)
	movl	$0, 4+mutex(%rip)
	movl	$0, 8+mutex(%rip)
	movl	$0, 12+mutex(%rip)
	movl	$0, 16+mutex(%rip)
	movw	$0, 20+mutex(%rip)
	movw	$0, 22+mutex(%rip)
	movq	$0, 24+mutex(%rip)
	movq	$0, 32+mutex(%rip)
	nop
.L2:
	movl	$0, -156(%rbp)
	jmp	.L3
.L4:
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	buffer(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -156(%rbp)
.L3:
	cmpl	$4, -156(%rbp)
	jle	.L4
	nop
.L5:
	movl	$0, out(%rip)
	nop
.L6:
	movl	$0, in(%rip)
	nop
.L7:
	movb	$0, full(%rip)
	movb	$0, 1+full(%rip)
	movb	$0, 2+full(%rip)
	movb	$0, 3+full(%rip)
	movb	$0, 4+full(%rip)
	movb	$0, 5+full(%rip)
	movb	$0, 6+full(%rip)
	movb	$0, 7+full(%rip)
	movb	$0, 8+full(%rip)
	movb	$0, 9+full(%rip)
	movb	$0, 10+full(%rip)
	movb	$0, 11+full(%rip)
	movb	$0, 12+full(%rip)
	movb	$0, 13+full(%rip)
	movb	$0, 14+full(%rip)
	movb	$0, 15+full(%rip)
	movb	$0, 16+full(%rip)
	movb	$0, 17+full(%rip)
	movb	$0, 18+full(%rip)
	movb	$0, 19+full(%rip)
	movb	$0, 20+full(%rip)
	movb	$0, 21+full(%rip)
	movb	$0, 22+full(%rip)
	movb	$0, 23+full(%rip)
	movb	$0, 24+full(%rip)
	movb	$0, 25+full(%rip)
	movb	$0, 26+full(%rip)
	movb	$0, 27+full(%rip)
	movb	$0, 28+full(%rip)
	movb	$0, 29+full(%rip)
	movb	$0, 30+full(%rip)
	movb	$0, 31+full(%rip)
	nop
.L8:
	movb	$0, empty(%rip)
	movb	$0, 1+empty(%rip)
	movb	$0, 2+empty(%rip)
	movb	$0, 3+empty(%rip)
	movb	$0, 4+empty(%rip)
	movb	$0, 5+empty(%rip)
	movb	$0, 6+empty(%rip)
	movb	$0, 7+empty(%rip)
	movb	$0, 8+empty(%rip)
	movb	$0, 9+empty(%rip)
	movb	$0, 10+empty(%rip)
	movb	$0, 11+empty(%rip)
	movb	$0, 12+empty(%rip)
	movb	$0, 13+empty(%rip)
	movb	$0, 14+empty(%rip)
	movb	$0, 15+empty(%rip)
	movb	$0, 16+empty(%rip)
	movb	$0, 17+empty(%rip)
	movb	$0, 18+empty(%rip)
	movb	$0, 19+empty(%rip)
	movb	$0, 20+empty(%rip)
	movb	$0, 21+empty(%rip)
	movb	$0, 22+empty(%rip)
	movb	$0, 23+empty(%rip)
	movb	$0, 24+empty(%rip)
	movb	$0, 25+empty(%rip)
	movb	$0, 26+empty(%rip)
	movb	$0, 27+empty(%rip)
	movb	$0, 28+empty(%rip)
	movb	$0, 29+empty(%rip)
	movb	$0, 30+empty(%rip)
	movb	$0, 31+empty(%rip)
	nop
.L9:
	movq	$0, _TIG_IZ_XFpM_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_XFpM_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_XFpM_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 211 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XFpM--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_XFpM_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_XFpM_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_XFpM_envp(%rip)
	nop
	movq	$6, -136(%rbp)
.L41:
	movq	-136(%rbp), %rax
	subq	$4, %rax
	cmpq	$23, %rax
	ja	.L44
	leaq	0(,%rax,4), %rdx
	leaq	.L16(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L16(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L16:
	.long	.L30-.L16
	.long	.L29-.L16
	.long	.L28-.L16
	.long	.L44-.L16
	.long	.L27-.L16
	.long	.L26-.L16
	.long	.L44-.L16
	.long	.L25-.L16
	.long	.L44-.L16
	.long	.L44-.L16
	.long	.L24-.L16
	.long	.L44-.L16
	.long	.L23-.L16
	.long	.L22-.L16
	.long	.L21-.L16
	.long	.L20-.L16
	.long	.L19-.L16
	.long	.L44-.L16
	.long	.L44-.L16
	.long	.L18-.L16
	.long	.L44-.L16
	.long	.L44-.L16
	.long	.L17-.L16
	.long	.L15-.L16
	.text
.L21:
	movl	$0, -140(%rbp)
	movq	$8, -136(%rbp)
	jmp	.L31
.L30:
	cmpl	$4, -148(%rbp)
	jg	.L32
	movq	$17, -136(%rbp)
	jmp	.L31
.L32:
	movq	$19, -136(%rbp)
	jmp	.L31
.L24:
	movl	$0, %esi
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_init@PLT
	movl	$5, %edx
	movl	$0, %esi
	leaq	empty(%rip), %rax
	movq	%rax, %rdi
	call	sem_init@PLT
	movl	$0, %edx
	movl	$0, %esi
	leaq	full(%rip), %rax
	movq	%rax, %rdi
	call	sem_init@PLT
	movl	$1, -128(%rbp)
	movl	$2, -124(%rbp)
	movl	$3, -120(%rbp)
	movl	$4, -116(%rbp)
	movl	$5, -112(%rbp)
	movl	$0, -152(%rbp)
	movq	$23, -136(%rbp)
	jmp	.L31
.L27:
	cmpl	$4, -140(%rbp)
	jg	.L34
	movq	$9, -136(%rbp)
	jmp	.L31
.L34:
	movq	$5, -136(%rbp)
	jmp	.L31
.L18:
	cmpl	$4, -152(%rbp)
	jg	.L36
	movq	$16, -136(%rbp)
	jmp	.L31
.L36:
	movq	$20, -136(%rbp)
	jmp	.L31
.L23:
	leaq	-128(%rbp), %rdx
	movl	-152(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	leaq	-96(%rbp), %rcx
	movl	-152(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rdx, %rcx
	leaq	producer(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	addl	$1, -152(%rbp)
	movq	$23, -136(%rbp)
	jmp	.L31
.L17:
	movl	-144(%rbp), %eax
	cltq
	movq	-96(%rbp,%rax,8), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	addl	$1, -144(%rbp)
	movq	$11, -136(%rbp)
	jmp	.L31
.L25:
	cmpl	$4, -144(%rbp)
	jg	.L38
	movq	$26, -136(%rbp)
	jmp	.L31
.L38:
	movq	$18, -136(%rbp)
	jmp	.L31
.L26:
	movl	-140(%rbp), %eax
	cltq
	movq	-48(%rbp,%rax,8), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	addl	$1, -140(%rbp)
	movq	$8, -136(%rbp)
	jmp	.L31
.L20:
	movl	$0, -144(%rbp)
	movq	$11, -136(%rbp)
	jmp	.L31
.L22:
	leaq	-128(%rbp), %rdx
	movl	-148(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	leaq	-48(%rbp), %rcx
	movl	-148(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rdx, %rcx
	leaq	consumer(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	addl	$1, -148(%rbp)
	movq	$4, -136(%rbp)
	jmp	.L31
.L28:
	movq	$14, -136(%rbp)
	jmp	.L31
.L15:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L42
	jmp	.L43
.L29:
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_destroy@PLT
	leaq	empty(%rip), %rax
	movq	%rax, %rdi
	call	sem_destroy@PLT
	leaq	full(%rip), %rax
	movq	%rax, %rdi
	call	sem_destroy@PLT
	movq	$27, -136(%rbp)
	jmp	.L31
.L19:
	movl	$0, -148(%rbp)
	movq	$4, -136(%rbp)
	jmp	.L31
.L44:
	nop
.L31:
	jmp	.L41
.L43:
	call	__stack_chk_fail@PLT
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.string	"Consumer %d: Remove Item %d from %d\n"
	.text
	.globl	consumer
	.type	consumer, @function
consumer:
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
	movq	$3, -8(%rbp)
.L55:
	cmpq	$4, -8(%rbp)
	je	.L46
	cmpq	$4, -8(%rbp)
	ja	.L57
	cmpq	$3, -8(%rbp)
	je	.L48
	cmpq	$3, -8(%rbp)
	ja	.L57
	cmpq	$1, -8(%rbp)
	je	.L49
	cmpq	$2, -8(%rbp)
	je	.L50
	jmp	.L57
.L46:
	movl	$0, %eax
	jmp	.L56
.L49:
	cmpl	$4, -16(%rbp)
	jg	.L52
	movq	$2, -8(%rbp)
	jmp	.L54
.L52:
	movq	$4, -8(%rbp)
	jmp	.L54
.L48:
	movl	$0, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L54
.L50:
	leaq	full(%rip), %rax
	movq	%rax, %rdi
	call	sem_wait@PLT
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_lock@PLT
	movl	out(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	buffer(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	out(%rip), %ecx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	-12(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	out(%rip), %eax
	leal	1(%rax), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, out(%rip)
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_unlock@PLT
	leaq	empty(%rip), %rax
	movq	%rax, %rdi
	call	sem_post@PLT
	addl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L54
.L57:
	nop
.L54:
	jmp	.L55
.L56:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	consumer, .-consumer
	.section	.rodata
	.align 8
.LC1:
	.string	"Producer %d: Insert Item %d at %d\n"
	.text
	.globl	producer
	.type	producer, @function
producer:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$6, -8(%rbp)
.L67:
	cmpq	$6, -8(%rbp)
	je	.L59
	cmpq	$6, -8(%rbp)
	ja	.L69
	cmpq	$5, -8(%rbp)
	je	.L61
	cmpq	$5, -8(%rbp)
	ja	.L69
	cmpq	$2, -8(%rbp)
	je	.L62
	cmpq	$3, -8(%rbp)
	jne	.L69
	call	rand@PLT
	movl	%eax, -12(%rbp)
	leaq	empty(%rip), %rax
	movq	%rax, %rdi
	call	sem_wait@PLT
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_lock@PLT
	movl	in(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	buffer(%rip), %rdx
	movl	-12(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movl	in(%rip), %ecx
	movl	in(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	buffer(%rip), %rax
	movl	(%rdx,%rax), %edx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	in(%rip), %eax
	leal	1(%rax), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, in(%rip)
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_unlock@PLT
	leaq	full(%rip), %rax
	movq	%rax, %rdi
	call	sem_post@PLT
	addl	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L63
.L59:
	movl	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L63
.L61:
	cmpl	$4, -16(%rbp)
	jg	.L64
	movq	$3, -8(%rbp)
	jmp	.L63
.L64:
	movq	$2, -8(%rbp)
	jmp	.L63
.L62:
	movl	$0, %eax
	jmp	.L68
.L69:
	nop
.L63:
	jmp	.L67
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	producer, .-producer
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
