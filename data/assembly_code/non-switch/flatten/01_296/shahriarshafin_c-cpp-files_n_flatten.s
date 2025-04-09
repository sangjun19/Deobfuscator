	.file	"shahriarshafin_c-cpp-files_n_flatten.c"
	.text
	.globl	_TIG_IZ_3G0G_argc
	.bss
	.align 4
	.type	_TIG_IZ_3G0G_argc, @object
	.size	_TIG_IZ_3G0G_argc, 4
_TIG_IZ_3G0G_argc:
	.zero	4
	.globl	_TIG_IZ_3G0G_envp
	.align 8
	.type	_TIG_IZ_3G0G_envp, @object
	.size	_TIG_IZ_3G0G_envp, 8
_TIG_IZ_3G0G_envp:
	.zero	8
	.globl	_TIG_IZ_3G0G_argv
	.align 8
	.type	_TIG_IZ_3G0G_argv, @object
	.size	_TIG_IZ_3G0G_argv, 8
_TIG_IZ_3G0G_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%s"
.LC1:
	.string	"1"
.LC2:
	.string	"4"
.LC3:
	.string	"?"
.LC4:
	.string	"-"
.LC5:
	.string	"+"
.LC6:
	.string	"%d"
.LC7:
	.string	"*"
.LC8:
	.string	"78"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2096, %rsp
	movl	%edi, -2068(%rbp)
	movq	%rsi, -2080(%rbp)
	movq	%rdx, -2088(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_3G0G_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_3G0G_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_3G0G_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 96 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3G0G--0
# 0 "" 2
#NO_APP
	movl	-2068(%rbp), %eax
	movl	%eax, _TIG_IZ_3G0G_argc(%rip)
	movq	-2080(%rbp), %rax
	movq	%rax, _TIG_IZ_3G0G_argv(%rip)
	movq	-2088(%rbp), %rax
	movq	%rax, _TIG_IZ_3G0G_envp(%rip)
	nop
	movq	$16, -2032(%rbp)
.L51:
	cmpq	$28, -2032(%rbp)
	ja	.L54
	movq	-2032(%rbp), %rax
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
	.long	.L30-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L54-.L8
	.long	.L23-.L8
	.long	.L54-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L54-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L54-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	movl	-2056(%rbp), %eax
	subl	$2, %eax
	cltq
	movzbl	-2016(%rbp,%rax), %eax
	cmpb	$51, %al
	jne	.L31
	movq	$26, -2032(%rbp)
	jmp	.L33
.L31:
	movq	$12, -2032(%rbp)
	jmp	.L33
.L28:
	leaq	-2016(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-2016(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2024(%rbp)
	movq	-2024(%rbp), %rax
	movl	%eax, -2056(%rbp)
	leaq	-2016(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2052(%rbp)
	movq	$28, -2032(%rbp)
	jmp	.L33
.L20:
	leaq	-2016(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2048(%rbp)
	movq	$8, -2032(%rbp)
	jmp	.L33
.L19:
	cmpl	$1, -2036(%rbp)
	jne	.L34
	movq	$24, -2032(%rbp)
	jmp	.L33
.L34:
	movq	$21, -2032(%rbp)
	jmp	.L33
.L22:
	movzbl	-2016(%rbp), %eax
	cmpb	$57, %al
	jne	.L36
	movq	$0, -2032(%rbp)
	jmp	.L33
.L36:
	movq	$22, -2032(%rbp)
	jmp	.L33
.L24:
	cmpl	$0, -2048(%rbp)
	jne	.L38
	movq	$19, -2032(%rbp)
	jmp	.L33
.L38:
	movq	$20, -2032(%rbp)
	jmp	.L33
.L29:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L18:
	movq	$10, -2032(%rbp)
	jmp	.L33
.L12:
	movl	-2060(%rbp), %eax
	movl	%eax, -2040(%rbp)
	movl	-2060(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -2060(%rbp)
	movq	$27, -2032(%rbp)
	jmp	.L33
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L52
	jmp	.L53
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L21:
	movl	-2056(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	-2016(%rbp,%rax), %eax
	cmpb	$53, %al
	jne	.L41
	movq	$25, -2032(%rbp)
	jmp	.L33
.L41:
	movq	$12, -2032(%rbp)
	jmp	.L33
.L16:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L17:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L26:
	cmpl	$0, -2044(%rbp)
	jne	.L43
	movq	$5, -2032(%rbp)
	jmp	.L33
.L43:
	movq	$13, -2032(%rbp)
	jmp	.L33
.L9:
	cmpl	$0, -2040(%rbp)
	je	.L45
	movq	$4, -2032(%rbp)
	jmp	.L33
.L45:
	movq	$10, -2032(%rbp)
	jmp	.L33
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L7:
	cmpl	$0, -2052(%rbp)
	jne	.L47
	movq	$17, -2032(%rbp)
	jmp	.L33
.L47:
	movq	$14, -2032(%rbp)
	jmp	.L33
.L27:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L23:
	leaq	-2060(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -2036(%rbp)
	movq	$15, -2032(%rbp)
	jmp	.L33
.L30:
	movl	-2056(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	-2016(%rbp,%rax), %eax
	cmpb	$52, %al
	jne	.L49
	movq	$7, -2032(%rbp)
	jmp	.L33
.L49:
	movq	$3, -2032(%rbp)
	jmp	.L33
.L25:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -2032(%rbp)
	jmp	.L33
.L15:
	leaq	-2016(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2044(%rbp)
	movq	$6, -2032(%rbp)
	jmp	.L33
.L54:
	nop
.L33:
	jmp	.L51
.L53:
	call	__stack_chk_fail@PLT
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
