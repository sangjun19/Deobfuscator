	.file	"HarrisonCamm_ENCE360_server_flatten.c"
	.text
	.globl	_TIG_IZ_qB6h_envp
	.bss
	.align 8
	.type	_TIG_IZ_qB6h_envp, @object
	.size	_TIG_IZ_qB6h_envp, 8
_TIG_IZ_qB6h_envp:
	.zero	8
	.globl	_TIG_IZ_qB6h_argc
	.align 4
	.type	_TIG_IZ_qB6h_argc, @object
	.size	_TIG_IZ_qB6h_argc, 4
_TIG_IZ_qB6h_argc:
	.zero	4
	.globl	_TIG_IZ_qB6h_argv
	.align 8
	.type	_TIG_IZ_qB6h_argv, @object
	.size	_TIG_IZ_qB6h_argv, 8
_TIG_IZ_qB6h_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"./mySocket"
.LC1:
	.string	"Connection accepted"
.LC2:
	.string	"%d"
.LC3:
	.string	"From client: %s\n"
.LC4:
	.string	"Hello to you too!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$736, %rsp
	movl	%edi, -708(%rbp)
	movq	%rsi, -720(%rbp)
	movq	%rdx, -728(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_qB6h_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qB6h_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qB6h_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 154 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qB6h--0
# 0 "" 2
#NO_APP
	movl	-708(%rbp), %eax
	movl	%eax, _TIG_IZ_qB6h_argc(%rip)
	movq	-720(%rbp), %rax
	movq	%rax, _TIG_IZ_qB6h_argv(%rip)
	movq	-728(%rbp), %rax
	movq	%rax, _TIG_IZ_qB6h_envp(%rip)
	nop
	movq	$9, -664(%rbp)
.L23:
	cmpq	$15, -664(%rbp)
	ja	.L26
	movq	-664(%rbp), %rax
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
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L26-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L7-.L8
	.text
.L14:
	cmpq	$0, -672(%rbp)
	jle	.L17
	movq	$5, -664(%rbp)
	jmp	.L19
.L17:
	movq	$11, -664(%rbp)
	jmp	.L19
.L7:
	cmpl	$511, -688(%rbp)
	jbe	.L20
	movq	$12, -664(%rbp)
	jmp	.L19
.L20:
	movq	$2, -664(%rbp)
	jmp	.L19
.L9:
	leaq	-528(%rbp), %rcx
	movl	-692(%rbp), %eax
	movl	$511, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -656(%rbp)
	movq	-656(%rbp), %rax
	movq	%rax, -672(%rbp)
	movq	$4, -664(%rbp)
	jmp	.L19
.L12:
	movl	$0, %edx
	movl	$5, %esi
	movl	$1, %edi
	call	socket@PLT
	movl	%eax, -684(%rbp)
	movl	-684(%rbp), %eax
	movl	%eax, -696(%rbp)
	leaq	-640(%rbp), %rax
	movl	$110, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$1, -640(%rbp)
	movq	$10, -648(%rbp)
	movq	-648(%rbp), %rax
	leaq	1(%rax), %rdx
	leaq	-640(%rbp), %rax
	addq	$2, %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	leaq	-640(%rbp), %rcx
	movl	-696(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	-696(%rbp), %eax
	movl	$10, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	-696(%rbp), %eax
	movl	$0, %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -680(%rbp)
	movl	-680(%rbp), %eax
	movl	%eax, -692(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-696(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movb	$0, -528(%rbp)
	movl	$1, -688(%rbp)
	movq	$15, -664(%rbp)
	jmp	.L19
.L15:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L24
	jmp	.L25
.L10:
	movl	-692(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-696(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movq	$3, -664(%rbp)
	jmp	.L19
.L11:
	movq	$8, -664(%rbp)
	jmp	.L19
.L13:
	leaq	-528(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-528(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movl	%eax, -676(%rbp)
	movl	-676(%rbp), %eax
	cltq
	movq	%rax, -672(%rbp)
	movq	-672(%rbp), %rdx
	leaq	-528(%rbp), %rcx
	movl	-692(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	-528(%rbp), %rcx
	movl	-692(%rbp), %eax
	movl	$511, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -672(%rbp)
	leaq	-528(%rbp), %rdx
	movq	-672(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$4, -664(%rbp)
	jmp	.L19
.L16:
	movl	-688(%rbp), %eax
	movb	$0, -528(%rbp,%rax)
	addl	$1, -688(%rbp)
	movq	$15, -664(%rbp)
	jmp	.L19
.L26:
	nop
.L19:
	jmp	.L23
.L25:
	call	__stack_chk_fail@PLT
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
