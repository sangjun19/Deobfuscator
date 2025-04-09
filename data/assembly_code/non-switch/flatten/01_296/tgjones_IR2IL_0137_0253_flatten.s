	.file	"tgjones_IR2IL_0137_0253_flatten.c"
	.text
	.globl	_TIG_IZ_0h7l_argc
	.bss
	.align 4
	.type	_TIG_IZ_0h7l_argc, @object
	.size	_TIG_IZ_0h7l_argc, 4
_TIG_IZ_0h7l_argc:
	.zero	4
	.globl	_TIG_IZ_0h7l_argv
	.align 8
	.type	_TIG_IZ_0h7l_argv, @object
	.size	_TIG_IZ_0h7l_argv, 8
_TIG_IZ_0h7l_argv:
	.zero	8
	.globl	_TIG_IZ_0h7l_envp
	.align 8
	.type	_TIG_IZ_0h7l_envp, @object
	.size	_TIG_IZ_0h7l_envp, 8
_TIG_IZ_0h7l_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"NG"
	.text
	.globl	fatal_error
	.type	fatal_error, @function
fatal_error:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	fatal_error, .-fatal_error
	.section	.rodata
.LC1:
	.string	"OK"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$0, _TIG_IZ_0h7l_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_0h7l_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_0h7l_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 112 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0h7l--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_0h7l_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_0h7l_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_0h7l_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L33:
	cmpq	$12, -8(%rbp)
	ja	.L34
	movq	-8(%rbp), %rax
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
	.long	.L25-.L16
	.long	.L24-.L16
	.long	.L34-.L16
	.long	.L34-.L16
	.long	.L23-.L16
	.long	.L22-.L16
	.long	.L34-.L16
	.long	.L21-.L16
	.long	.L20-.L16
	.long	.L19-.L16
	.long	.L18-.L16
	.long	.L17-.L16
	.long	.L15-.L16
	.text
.L23:
	movl	$0, %edi
	call	exit@PLT
.L15:
	subl	$1, -60(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L26
.L20:
	movl	-60(%rbp), %eax
	cmpl	-56(%rbp), %eax
	jge	.L27
	movq	$5, -8(%rbp)
	jmp	.L26
.L27:
	movq	$9, -8(%rbp)
	jmp	.L26
.L24:
	movl	$0, -52(%rbp)
	movl	$-2, -48(%rbp)
	movl	$-1, -44(%rbp)
	movl	$0, -40(%rbp)
	movl	$1, -36(%rbp)
	movl	$5, -32(%rbp)
	movl	$6, -28(%rbp)
	movl	$8, -24(%rbp)
	movl	$9, -20(%rbp)
	movl	$10, -16(%rbp)
	movl	$11, -12(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, -56(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, -60(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L26
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L26
.L19:
	cmpl	$0, -60(%rbp)
	je	.L29
	movq	$10, -8(%rbp)
	jmp	.L26
.L29:
	movq	$11, -8(%rbp)
	jmp	.L26
.L22:
	cmpl	$5, -60(%rbp)
	jne	.L31
	movq	$7, -8(%rbp)
	jmp	.L26
.L31:
	movq	$12, -8(%rbp)
	jmp	.L26
.L18:
	call	fatal_error
	movq	$4, -8(%rbp)
	jmp	.L26
.L25:
	movq	$1, -8(%rbp)
	jmp	.L26
.L21:
	call	fatal_error
	movq	$12, -8(%rbp)
	jmp	.L26
.L34:
	nop
.L26:
	jmp	.L33
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
