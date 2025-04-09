	.file	"XiyouyLinuxGroup-2017-Summer_teamA_rep_fork_flatten.c"
	.text
	.globl	_TIG_IZ_UHGW_argc
	.bss
	.align 4
	.type	_TIG_IZ_UHGW_argc, @object
	.size	_TIG_IZ_UHGW_argc, 4
_TIG_IZ_UHGW_argc:
	.zero	4
	.globl	_TIG_IZ_UHGW_envp
	.align 8
	.type	_TIG_IZ_UHGW_envp, @object
	.size	_TIG_IZ_UHGW_envp, 8
_TIG_IZ_UHGW_envp:
	.zero	8
	.globl	_TIG_IZ_UHGW_argv
	.align 8
	.type	_TIG_IZ_UHGW_argv, @object
	.size	_TIG_IZ_UHGW_argv, 8
_TIG_IZ_UHGW_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Child running"
.LC1:
	.string	"Parent running"
.LC2:
	.string	"process creation %d,%d\n"
.LC3:
	.string	"failed\n"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_UHGW_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_UHGW_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_UHGW_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-UHGW--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_UHGW_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_UHGW_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_UHGW_envp(%rip)
	nop
	movq	$6, -8(%rbp)
.L25:
	cmpq	$12, -8(%rbp)
	ja	.L28
	movq	-8(%rbp), %rax
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, -16(%rbp)
	movl	$5, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L17
.L14:
	movl	$3, -32(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L17
.L9:
	cmpl	$0, -32(%rbp)
	jle	.L18
	movq	$10, -8(%rbp)
	jmp	.L17
.L18:
	movq	$2, -8(%rbp)
	jmp	.L17
.L11:
	cmpl	$-1, -28(%rbp)
	je	.L20
	cmpl	$0, -28(%rbp)
	je	.L21
	jmp	.L26
.L20:
	movq	$7, -8(%rbp)
	jmp	.L23
.L21:
	movq	$12, -8(%rbp)
	jmp	.L23
.L26:
	movq	$3, -8(%rbp)
	nop
.L23:
	jmp	.L17
.L13:
	movq	$0, -8(%rbp)
	jmp	.L17
.L10:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	sleep@PLT
	subl	$1, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L17
.L16:
	call	getppid@PLT
	movl	%eax, -24(%rbp)
	call	getpid@PLT
	movl	%eax, -20(%rbp)
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	fork@PLT
	movl	%eax, -28(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L17
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -8(%rbp)
	jmp	.L17
.L15:
	movl	$0, %eax
	jmp	.L27
.L28:
	nop
.L17:
	jmp	.L25
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
