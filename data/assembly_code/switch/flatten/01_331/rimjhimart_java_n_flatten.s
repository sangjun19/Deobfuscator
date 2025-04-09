	.file	"rimjhimart_java_n_flatten.c"
	.text
	.globl	_TIG_IZ_nSwV_argv
	.bss
	.align 8
	.type	_TIG_IZ_nSwV_argv, @object
	.size	_TIG_IZ_nSwV_argv, 8
_TIG_IZ_nSwV_argv:
	.zero	8
	.globl	_TIG_IZ_nSwV_argc
	.align 4
	.type	_TIG_IZ_nSwV_argc, @object
	.size	_TIG_IZ_nSwV_argc, 4
_TIG_IZ_nSwV_argc:
	.zero	4
	.globl	_TIG_IZ_nSwV_envp
	.align 8
	.type	_TIG_IZ_nSwV_envp, @object
	.size	_TIG_IZ_nSwV_envp, 8
_TIG_IZ_nSwV_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"saturday"
.LC1:
	.string	"wednesday"
.LC2:
	.string	"friday"
.LC3:
	.string	"enter the number 1 to 7"
.LC4:
	.string	"%d"
.LC5:
	.string	"thursday"
.LC6:
	.string	"monday"
.LC7:
	.string	"tuesday"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_nSwV_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_nSwV_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_nSwV_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-nSwV--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_nSwV_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_nSwV_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_nSwV_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L30:
	cmpq	$18, -16(%rbp)
	ja	.L33
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L33-.L8
	.long	.L16-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L15-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L14-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L34-.L8
	.long	.L10-.L8
	.long	.L33-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L17:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L19
.L16:
	movq	$1, -16(%rbp)
	jmp	.L19
.L14:
	movl	-20(%rbp), %eax
	cmpl	$6, %eax
	ja	.L21
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L21-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L22:
	movq	$18, -16(%rbp)
	jmp	.L29
.L24:
	movq	$12, -16(%rbp)
	jmp	.L29
.L25:
	movq	$17, -16(%rbp)
	jmp	.L29
.L26:
	movq	$15, -16(%rbp)
	jmp	.L29
.L27:
	movq	$0, -16(%rbp)
	jmp	.L29
.L28:
	movq	$6, -16(%rbp)
	jmp	.L29
.L21:
	movq	$13, -16(%rbp)
	nop
.L29:
	jmp	.L19
.L12:
	movq	$14, -16(%rbp)
	jmp	.L19
.L9:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L15:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L18:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L19
.L33:
	nop
.L19:
	jmp	.L30
.L34:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L32
	call	__stack_chk_fail@PLT
.L32:
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
