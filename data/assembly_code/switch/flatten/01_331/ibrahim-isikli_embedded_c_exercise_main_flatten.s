	.file	"ibrahim-isikli_embedded_c_exercise_main_flatten.c"
	.text
	.globl	_TIG_IZ_x4P8_envp
	.bss
	.align 8
	.type	_TIG_IZ_x4P8_envp, @object
	.size	_TIG_IZ_x4P8_envp, 8
_TIG_IZ_x4P8_envp:
	.zero	8
	.globl	_TIG_IZ_x4P8_argv
	.align 8
	.type	_TIG_IZ_x4P8_argv, @object
	.size	_TIG_IZ_x4P8_argv, 8
_TIG_IZ_x4P8_argv:
	.zero	8
	.globl	_TIG_IZ_x4P8_argc
	.align 4
	.type	_TIG_IZ_x4P8_argc, @object
	.size	_TIG_IZ_x4P8_argc, 4
_TIG_IZ_x4P8_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"invalid otion!"
.LC1:
	.string	"result:\t%d"
	.align 8
.LC2:
	.string	"[1]sum\n[2]subtract\n[3]multiply"
.LC3:
	.string	"choose your option\t"
.LC4:
	.string	"%d"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_x4P8_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_x4P8_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_x4P8_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-x4P8--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_x4P8_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_x4P8_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_x4P8_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L25:
	cmpq	$13, -16(%rbp)
	ja	.L28
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
	.long	.L28-.L8
	.long	.L17-.L8
	.long	.L28-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L18
.L12:
	leaq	multiply(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L18
.L17:
	leaq	subtract(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L18
.L16:
	leaq	add(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L18
.L9:
	movl	-36(%rbp), %eax
	cmpl	$3, %eax
	je	.L19
	cmpl	$3, %eax
	jg	.L20
	cmpl	$1, %eax
	je	.L21
	cmpl	$2, %eax
	je	.L22
	jmp	.L20
.L19:
	movq	$8, -16(%rbp)
	jmp	.L23
.L22:
	movq	$1, -16(%rbp)
	jmp	.L23
.L21:
	movq	$3, -16(%rbp)
	jmp	.L23
.L20:
	movq	$4, -16(%rbp)
	nop
.L23:
	jmp	.L18
.L11:
	movq	-24(%rbp), %rax
	movl	$2, %esi
	movl	$4, %edi
	call	*%rax
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L18
.L7:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -16(%rbp)
	jmp	.L18
.L14:
	movl	$1, %eax
	jmp	.L26
.L10:
	movl	$0, %eax
	jmp	.L26
.L13:
	movq	$13, -16(%rbp)
	jmp	.L18
.L28:
	nop
.L18:
	jmp	.L25
.L26:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	call	__stack_chk_fail@PLT
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	subtract
	.type	subtract, @function
subtract:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L32:
	cmpq	$0, -8(%rbp)
	jne	.L35
	movl	-20(%rbp), %eax
	subl	-24(%rbp), %eax
	jmp	.L34
.L35:
	nop
	jmp	.L32
.L34:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	subtract, .-subtract
	.globl	add
	.type	add, @function
add:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L39:
	cmpq	$0, -8(%rbp)
	jne	.L42
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	jmp	.L41
.L42:
	nop
	jmp	.L39
.L41:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	add, .-add
	.globl	multiply
	.type	multiply, @function
multiply:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L46:
	cmpq	$0, -8(%rbp)
	jne	.L49
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	jmp	.L48
.L49:
	nop
	jmp	.L46
.L48:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	multiply, .-multiply
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
