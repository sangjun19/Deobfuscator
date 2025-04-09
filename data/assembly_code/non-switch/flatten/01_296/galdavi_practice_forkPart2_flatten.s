	.file	"galdavi_practice_forkPart2_flatten.c"
	.text
	.globl	_TIG_IZ_fGGP_envp
	.bss
	.align 8
	.type	_TIG_IZ_fGGP_envp, @object
	.size	_TIG_IZ_fGGP_envp, 8
_TIG_IZ_fGGP_envp:
	.zero	8
	.globl	_TIG_IZ_fGGP_argv
	.align 8
	.type	_TIG_IZ_fGGP_argv, @object
	.size	_TIG_IZ_fGGP_argv, 8
_TIG_IZ_fGGP_argv:
	.zero	8
	.globl	_TIG_IZ_fGGP_argc
	.align 4
	.type	_TIG_IZ_fGGP_argc, @object
	.size	_TIG_IZ_fGGP_argc, 4
_TIG_IZ_fGGP_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"David Solis Gallo (PID%d)\n"
	.align 8
.LC1:
	.string	"Parent process has completed its work."
	.align 8
.LC2:
	.string	"Child process has exited with status %d.\n"
.LC3:
	.string	"Arcane (PID%d)\n"
	.align 8
.LC4:
	.string	"Child process has completed its work."
.LC5:
	.string	"fork error.\n"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_fGGP_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_fGGP_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_fGGP_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fGGP--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_fGGP_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_fGGP_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_fGGP_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L22:
	cmpq	$9, -16(%rbp)
	ja	.L25
	movq	-16(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$0, %eax
	jmp	.L23
.L14:
	call	fork@PLT
	movl	%eax, -28(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L17
.L7:
	call	getpid@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$3, %edi
	call	sleep@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rcx
	movl	-28(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movl	-32(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L11:
	movl	$1, %eax
	jmp	.L23
.L12:
	cmpl	$0, -28(%rbp)
	jns	.L18
	movq	$2, -16(%rbp)
	jmp	.L17
.L18:
	movq	$0, -16(%rbp)
	jmp	.L17
.L15:
	cmpl	$0, -28(%rbp)
	jne	.L20
	movq	$7, -16(%rbp)
	jmp	.L17
.L20:
	movq	$9, -16(%rbp)
	jmp	.L17
.L10:
	call	getpid@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	sleep@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$6, -16(%rbp)
	jmp	.L17
.L25:
	nop
.L17:
	jmp	.L22
.L23:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L24
	call	__stack_chk_fail@PLT
.L24:
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
