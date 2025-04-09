	.file	"GuilhermeMoura5151_SO_fork3_flatten.c"
	.text
	.globl	_TIG_IZ_bzde_envp
	.bss
	.align 8
	.type	_TIG_IZ_bzde_envp, @object
	.size	_TIG_IZ_bzde_envp, 8
_TIG_IZ_bzde_envp:
	.zero	8
	.globl	_TIG_IZ_bzde_argv
	.align 8
	.type	_TIG_IZ_bzde_argv, @object
	.size	_TIG_IZ_bzde_argv, 8
_TIG_IZ_bzde_argv:
	.zero	8
	.globl	_TIG_IZ_bzde_argc
	.align 4
	.type	_TIG_IZ_bzde_argc, @object
	.size	_TIG_IZ_bzde_argc, 4
_TIG_IZ_bzde_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"erro no lancamento da aplicacao"
	.align 8
.LC1:
	.string	"Pai (depois do fork): PID = %u, PPID = %u\n"
.LC2:
	.string	"Erro no fork\n"
.LC3:
	.string	"-l"
.LC4:
	.string	"ls"
.LC5:
	.string	"/bin/ls"
	.align 8
.LC6:
	.string	"Porque \303\251 que eu n\303\243o apare\303\247o?"
	.align 8
.LC7:
	.string	"Pai (antes do fork): PID = %u, PPID = %u\n"
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
	movq	$0, _TIG_IZ_bzde_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_bzde_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_bzde_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 115 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-bzde--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_bzde_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_bzde_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_bzde_envp(%rip)
	nop
	movq	$8, -8(%rbp)
.L27:
	cmpq	$13, -8(%rbp)
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
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L28-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L16:
	movl	$0, %eax
	jmp	.L20
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$10, -8(%rbp)
	jmp	.L21
.L12:
	movq	$7, -8(%rbp)
	jmp	.L21
.L18:
	movl	$1, %edi
	call	sleep@PLT
	call	getppid@PLT
	movl	%eax, -16(%rbp)
	call	getpid@PLT
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L21
.L10:
	cmpl	$-1, -32(%rbp)
	je	.L22
	cmpl	$0, -32(%rbp)
	jne	.L23
	movq	$5, -8(%rbp)
	jmp	.L24
.L22:
	movq	$13, -8(%rbp)
	jmp	.L24
.L23:
	movq	$1, -8(%rbp)
	nop
.L24:
	jmp	.L21
.L7:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$6, -8(%rbp)
	jmp	.L21
.L14:
	movl	$1, %eax
	jmp	.L20
.L15:
	movl	$0, %ecx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execl@PLT
	movl	%eax, -28(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L21
.L11:
	movl	$1, %eax
	jmp	.L20
.L19:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L21
.L13:
	call	getppid@PLT
	movl	%eax, -24(%rbp)
	call	getpid@PLT
	movl	%eax, -20(%rbp)
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	fork@PLT
	movl	%eax, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L21
.L17:
	cmpl	$0, -28(%rbp)
	jns	.L25
	movq	$12, -8(%rbp)
	jmp	.L21
.L25:
	movq	$0, -8(%rbp)
	jmp	.L21
.L28:
	nop
.L21:
	jmp	.L27
.L20:
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
